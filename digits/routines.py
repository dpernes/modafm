import time
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from models import MDANet, MixMDANet, GradientReversalLayer
from augment import RandAugment
from utils import fixmatch_loss, save_checkpoint


def mdan_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    model = MDANet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for epoch in range(cfg['epochs']):
        print('Epoch {}/{}'.format(epoch, cfg['epochs']-1))
        model.train()
        loss_hist = []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_losses_hist = [[] for i in range(len(train_loader.sources))]
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)
            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            class_losses = torch.stack([F.cross_entropy(ys_logits[i], ys[i]) for i in range(len(train_loader.sources))])
            domain_losses = torch.stack([F.cross_entropy(ds_logits[i], ds[i]) + F.cross_entropy(dt_logits[i], dt)
                                         for i in range(len(train_loader.sources))])
            if cfg['mode'] == 'dynamic':
                loss = torch.log(torch.sum(torch.exp(cfg['gamma']*(class_losses + cfg['mu']*domain_losses))))/cfg['gamma']
            else:
                loss = torch.max(class_losses + cfg['mu']*domain_losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                domain_losses_hist[j].append(domain_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        print()
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_losses = [sum(domain_losses_hist[i])/len(domain_losses_hist[i]) for i in range(len(train_loader.sources))]
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        print('global loss: {:.3f} | class losses: [{:.3f}, {:.3f}, {:.3f}] | domain losses: [{:.3f}, {:.3f}, {:.3f}]'.format(
            loss, class_losses[0], class_losses[1], class_losses[2], domain_losses[0], domain_losses[1], domain_losses[2]))
        print('{:.1f} seconds'.format(t1-t0))
        print()
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', cfg['mode'], epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        # EVALUATE ON TEST SET --> CHEATING :D
        model.eval()
        for split_name, test_loader in zip(['pub target', 'priv target'], [test_pub_loader, test_priv_loader]):
            if test_loader is None:
                continue
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(test_loader):
                Xt, yt = Xt.float().to(device), yt.long().to(device)
                with torch.no_grad():
                    yt_logits = model.inference(Xt)
                loss = F.cross_entropy(yt_logits, yt)
                yt_preds = torch.argmax(yt_logits, dim=1)
                acc = torch.mean((yt_preds == yt).float())
                loss_hist.append(loss.item())
                acc_hist.append(acc.item())
            loss = sum(loss_hist)/len(loss_hist)
            acc = sum(acc_hist)/len(acc_hist)
            if cfg['use_visdom']:
                loss_plt.plot('class loss', cfg['target']+' '+split_name, 'XEN', epoch, loss)
                loss_plt.plot('accuracy', cfg['target']+' '+split_name, 'ACC', epoch, acc)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

    return model

def mdan_unif_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    model = MDANet().to(device)
    model.grad_reverse = nn.ModuleList([nn.Identity() for _ in range(len(model.domain_class))])  # remove grad reverse
    task_optim = optim.Adadelta(list(model.feat_ext.parameters())+list(model.task_class.parameters()),
                                lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    adv_optim = optim.Adadelta(model.domain_class.parameters(),
                               lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for epoch in range(cfg['epochs']):
        print('Epoch {}/{}'.format(epoch, cfg['epochs']-1))
        model.train()
        loss_hist = []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_losses_hist = [[] for i in range(len(train_loader.sources))]
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)
            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            class_losses = torch.stack([F.cross_entropy(ys_logits[i], ys[i]) for i in range(len(train_loader.sources))])
            domain_losses = torch.stack([F.cross_entropy(ds_logits[i], ds[i]) + F.cross_entropy(dt_logits[i], dt)
                                         for i in range(len(train_loader.sources))])
            unif_losses = -torch.stack([torch.mean(F.log_softmax(ds_logits[i], dim=1)) + torch.mean(F.log_softmax(dt_logits[i], dim=1))
                                        for i in range(len(train_loader.sources))])
            if cfg['mode'] == 'dynamic':
                loss = torch.log(torch.sum(torch.exp(cfg['gamma']*(class_losses + cfg['mu']*unif_losses))))/cfg['gamma']
                adv_loss = torch.sum(domain_losses)
            else:
                loss = torch.max(class_losses + cfg['mu']*unif_losses)
                adv_loss = torch.sum(domain_losses)

            adv_optim.zero_grad()
            task_optim.zero_grad()
            adv_loss.backward(retain_graph=True)
            adv_optim.step()

            adv_optim.zero_grad()
            task_optim.zero_grad()
            loss.backward()
            task_optim.step()

            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                domain_losses_hist[j].append(domain_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        print()
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_losses = [sum(domain_losses_hist[i])/len(domain_losses_hist[i]) for i in range(len(train_loader.sources))]
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        print('global loss: {:.3f} | class losses: [{:.3f}, {:.3f}, {:.3f}] | domain losses: [{:.3f}, {:.3f}, {:.3f}]'.format(
            loss, class_losses[0], class_losses[1], class_losses[2], domain_losses[0], domain_losses[1], domain_losses[2]))
        print('{:.1f} seconds'.format(t1-t0))
        print()
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', cfg['mode'], epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        # EVALUATE ON TEST SET --> CHEATING :D
        model.eval()
        for split_name, test_loader in zip(['pub target', 'priv target'], [test_pub_loader, test_priv_loader]):
            if test_loader is None:
                continue
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(test_loader):
                Xt, yt = Xt.float().to(device), yt.long().to(device)
                with torch.no_grad():
                    yt_logits = model.inference(Xt)
                loss = F.cross_entropy(yt_logits, yt)
                yt_preds = torch.argmax(yt_logits, dim=1)
                acc = torch.mean((yt_preds == yt).float())
                loss_hist.append(loss.item())
                acc_hist.append(acc.item())
            loss = sum(loss_hist)/len(loss_hist)
            acc = sum(acc_hist)/len(acc_hist)
            if cfg['use_visdom']:
                loss_plt.plot('class loss', cfg['target']+' '+split_name, 'XEN', epoch, loss)
                loss_plt.plot('accuracy', cfg['target']+' '+split_name, 'ACC', epoch, acc)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, [task_optim, adv_optim], None, name=cfg['output']+'.ckp'+str(epoch))

    return model

def mdan_fm_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    model = MDANet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for epoch in range(cfg['epochs']):
        print('Epoch {}/{}'.format(epoch, cfg['epochs']-1))
        model.train()
        loss_hist = []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_losses_hist = [[] for i in range(len(train_loader.sources))]
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)

            m_aug = np.random.randint(cfg['m_min_rand_aug'], cfg['m_max_rand_aug']+1)
            aug_transf = lambda batch: torch.stack([RandAugment(cfg['n_rand_aug'], m_aug)(img) for img in batch])
            Xt_aug = aug_transf(Xt.cpu()).to(device)

            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            yt_logits = model.inference(Xt)
            yt_aug_logits = model.inference(Xt_aug)

            class_losses = torch.stack([F.cross_entropy(ys_logits[i], ys[i]) for i in range(len(train_loader.sources))])
            domain_losses = torch.stack([F.cross_entropy(ds_logits[i], ds[i]) + F.cross_entropy(dt_logits[i], dt)
                                         for i in range(len(train_loader.sources))])
            fm_loss = fixmatch_loss(yt_logits, yt_aug_logits)
            if cfg['mode'] == 'dynamic':
                loss = torch.log(torch.sum(torch.exp(cfg['gamma']*(class_losses + cfg['mu']*unif_losses))))/cfg['gamma']
            else:
                loss = torch.max(class_losses + cfg['mu']*domain_losses)
            loss += cfg['lambda']*fm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                domain_losses_hist[j].append(domain_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        print()
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_losses = [sum(domain_losses_hist[i])/len(domain_losses_hist[i]) for i in range(len(train_loader.sources))]
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        print('global loss: {:.3f} | class losses: [{:.3f}, {:.3f}, {:.3f}] | domain losses: [{:.3f}, {:.3f}, {:.3f}]'.format(
            loss, class_losses[0], class_losses[1], class_losses[2], domain_losses[0], domain_losses[1], domain_losses[2]))
        print('{:.1f} seconds'.format(t1-t0))
        print()
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', cfg['mode'], epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        # EVALUATE ON TEST SET --> CHEATING :D
        model.eval()
        for split_name, test_loader in zip(['pub target', 'priv target'], [test_pub_loader, test_priv_loader]):
            if test_loader is None:
                continue
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(test_loader):
                Xt, yt = Xt.float().to(device), yt.long().to(device)
                with torch.no_grad():
                    yt_logits = model.inference(Xt)
                loss = F.cross_entropy(yt_logits, yt)
                yt_preds = torch.argmax(yt_logits, dim=1)
                acc = torch.mean((yt_preds == yt).float())
                loss_hist.append(loss.item())
                acc_hist.append(acc.item())
            loss = sum(loss_hist)/len(loss_hist)
            acc = sum(acc_hist)/len(acc_hist)
            if cfg['use_visdom']:
                loss_plt.plot('class loss', cfg['target']+' '+split_name, 'XEN', epoch, loss)
                loss_plt.plot('accuracy', cfg['target']+' '+split_name, 'ACC', epoch, acc)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

    return model

def mdan_unif_fm_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    model = MDANet().to(device)
    model.grad_reverse = nn.ModuleList([nn.Identity() for _ in range(len(model.domain_class))])  # remove grad reverse
    task_optim = optim.Adadelta(list(model.feat_ext.parameters())+list(model.task_class.parameters()),
                                lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    adv_optim = optim.Adadelta(model.domain_class.parameters(),
                               lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for epoch in range(cfg['epochs']):
        print('Epoch {}/{}'.format(epoch, cfg['epochs']-1))
        model.train()
        loss_hist = []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_losses_hist = [[] for i in range(len(train_loader.sources))]
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)

            m_aug = np.random.randint(cfg['m_min_rand_aug'], cfg['m_max_rand_aug']+1)
            aug_transf = lambda batch: torch.stack([RandAugment(cfg['n_rand_aug'], m_aug)(img) for img in batch])
            Xt_aug = aug_transf(Xt.cpu()).to(device)

            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            yt_logits = model.inference(Xt)
            yt_aug_logits = model.inference(Xt_aug)

            class_losses = torch.stack([F.cross_entropy(ys_logits[i], ys[i]) for i in range(len(train_loader.sources))])
            domain_losses = torch.stack([F.cross_entropy(ds_logits[i], ds[i]) + F.cross_entropy(dt_logits[i], dt)
                                         for i in range(len(train_loader.sources))])
            unif_losses = -torch.stack([torch.mean(F.log_softmax(ds_logits[i], dim=1)) + torch.mean(F.log_softmax(dt_logits[i], dim=1))
                                        for i in range(len(train_loader.sources))])
            fm_loss = fixmatch_loss(yt_logits, yt_aug_logits)
            if cfg['mode'] == 'dynamic':
                loss = torch.log(torch.sum(torch.exp(cfg['gamma']*(class_losses + cfg['mu']*unif_losses))))/cfg['gamma']
                adv_loss = torch.sum(domain_losses)
            else:
                loss = torch.max(class_losses + cfg['mu']*domain_losses)
                adv_loss = torch.sum(domain_losses)

            loss += cfg['lambda']*fm_loss

            adv_optim.zero_grad()
            task_optim.zero_grad()
            adv_loss.backward(retain_graph=True)
            adv_optim.step()

            adv_optim.zero_grad()
            task_optim.zero_grad()
            loss.backward()
            task_optim.step()

            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                domain_losses_hist[j].append(domain_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        print()
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_losses = [sum(domain_losses_hist[i])/len(domain_losses_hist[i]) for i in range(len(train_loader.sources))]
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        print('global loss: {:.3f} | class losses: [{:.3f}, {:.3f}, {:.3f}] | domain losses: [{:.3f}, {:.3f}, {:.3f}]'.format(
            loss, class_losses[0], class_losses[1], class_losses[2], domain_losses[0], domain_losses[1], domain_losses[2]))
        print('{:.1f} seconds'.format(t1-t0))
        print()
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', cfg['mode'], epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        # EVALUATE ON TEST SET --> CHEATING :D
        model.eval()
        for split_name, test_loader in zip(['pub target', 'priv target'], [test_pub_loader, test_priv_loader]):
            if test_loader is None:
                continue
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(test_loader):
                Xt, yt = Xt.float().to(device), yt.long().to(device)
                with torch.no_grad():
                    yt_logits = model.inference(Xt)
                loss = F.cross_entropy(yt_logits, yt)
                yt_preds = torch.argmax(yt_logits, dim=1)
                acc = torch.mean((yt_preds == yt).float())
                loss_hist.append(loss.item())
                acc_hist.append(acc.item())
            loss = sum(loss_hist)/len(loss_hist)
            acc = sum(acc_hist)/len(acc_hist)
            if cfg['use_visdom']:
                loss_plt.plot('class loss', cfg['target']+' '+split_name, 'XEN', epoch, loss)
                loss_plt.plot('accuracy', cfg['target']+' '+split_name, 'ACC', epoch, acc)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, [task_optim, adv_optim], None, name=cfg['output']+'.ckp'+str(epoch))

    return model

def mixmdan_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    model = MixMDANet().to(device)
    alpha = nn.Parameter(torch.Tensor(len(train_loader.sources)).to(device))
    nn.init.uniform_(alpha)
    grad_reverse_fn = GradientReversalLayer().to(device)

    optimizer = optim.Adadelta(list(model.parameters()) + [alpha], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for epoch in range(cfg['epochs']):
        print('Epoch {}/{}'.format(epoch, cfg['epochs']-1))
        model.train()
        loss_hist = []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_loss_hist = []
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)
            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            class_losses = torch.stack([F.cross_entropy(ys_logits[i], ys[i]) for i in range(len(train_loader.sources))])
            src_domain_losses = torch.stack([F.cross_entropy(ds_logits[i], ds[i])
                                             for i in range(len(train_loader.sources))])
            tgt_domain_loss = F.cross_entropy(dt_logits, dt)

            beta = F.softmax(alpha, dim=0)
            beta_grad_rev = grad_reverse_fn(beta)
            mix_domain_loss = torch.sum(beta_grad_rev * src_domain_losses) + tgt_domain_loss
            mix_class_loss = torch.sum(beta * class_losses)
            loss = mix_class_loss + cfg['mu']*mix_domain_loss + cfg['beta']*torch.sum(beta**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())
            domain_loss_hist.append(mix_domain_loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        print()
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_loss = sum(domain_loss_hist)/len(domain_loss_hist)
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        print('global loss: {:.3f} | class losses: [{:.3f}, {:.3f}, {:.3f}] | domain loss: {:.3f}'.format(
            loss, class_losses[0], class_losses[1], class_losses[2], domain_loss))
        print('mix coef: [{:.3f}, {:.3f}, {:.3f}]'.format(beta[0], beta[1], beta[2]))
        print('{:.1f} seconds'.format(t1-t0))
        print()
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', cfg['mode'], epoch, loss)
            loss_plt.plot('domain loss', 'train', 'XEN', epoch, domain_loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        # EVALUATE ON TEST SET --> CHEATING :D
        model.eval()
        for split_name, test_loader in zip(['pub target', 'priv target'], [test_pub_loader, test_priv_loader]):
            if test_loader is None:
                continue
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(test_loader):
                Xt, yt = Xt.float().to(device), yt.long().to(device)
                with torch.no_grad():
                    yt_logits = model.inference(Xt)
                loss = F.cross_entropy(yt_logits, yt)
                yt_preds = torch.argmax(yt_logits, dim=1)
                acc = torch.mean((yt_preds == yt).float())
                loss_hist.append(loss.item())
                acc_hist.append(acc.item())
            loss = sum(loss_hist)/len(loss_hist)
            acc = sum(acc_hist)/len(acc_hist)
            if cfg['use_visdom']:
                loss_plt.plot('class loss', cfg['target']+' '+split_name, 'XEN', epoch, loss)
                loss_plt.plot('accuracy', cfg['target']+' '+split_name, 'ACC', epoch, acc)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

    return model

def mixmdan_fm_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    model = MixMDANet().to(device)
    alpha = nn.Parameter(torch.Tensor(len(train_loader.sources)).to(device))
    nn.init.uniform_(alpha)
    grad_reverse_fn = GradientReversalLayer().to(device)

    optimizer = optim.Adadelta(list(model.parameters()) + [alpha], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for epoch in range(cfg['epochs']):
        print('Epoch {}/{}'.format(epoch, cfg['epochs']-1))
        model.train()
        loss_hist = []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_loss_hist = []
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)

            m_aug = np.random.randint(cfg['m_min_rand_aug'], cfg['m_max_rand_aug']+1)
            aug_transf = lambda batch: torch.stack([RandAugment(cfg['n_rand_aug'], m_aug)(img) for img in batch])
            Xt_aug = aug_transf(Xt.cpu()).to(device)

            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            yt_logits = model.inference(Xt)
            yt_aug_logits = model.inference(Xt_aug)

            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            class_losses = torch.stack([F.cross_entropy(ys_logits[i], ys[i]) for i in range(len(train_loader.sources))])
            src_domain_losses = torch.stack([F.cross_entropy(ds_logits[i], ds[i])
                                             for i in range(len(train_loader.sources))])
            tgt_domain_loss = F.cross_entropy(dt_logits, dt)
            fm_loss = fixmatch_loss(yt_logits, yt_aug_logits)

            beta = F.softmax(alpha, dim=0)
            beta_grad_rev = grad_reverse_fn(beta)
            mix_domain_loss = torch.sum(beta_grad_rev * src_domain_losses) + tgt_domain_loss
            mix_class_loss = torch.sum(beta * class_losses)
            loss = mix_class_loss + cfg['mu']*mix_domain_loss + cfg['beta']*torch.sum(beta**2) + cfg['lambda']*fm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())
            domain_loss_hist.append(mix_domain_loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        print()
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_loss = sum(domain_loss_hist)/len(domain_loss_hist)
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        print('global loss: {:.3f} | class losses: [{:.3f}, {:.3f}, {:.3f}] | domain loss: {:.3f}'.format(
            loss, class_losses[0], class_losses[1], class_losses[2], domain_loss))
        print('mix coef: [{:.3f}, {:.3f}, {:.3f}]'.format(beta[0], beta[1], beta[2]))
        print('{:.1f} seconds'.format(t1-t0))
        print()
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', cfg['mode'], epoch, loss)
            loss_plt.plot('domain loss', 'train', 'XEN', epoch, domain_loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        # EVALUATE ON TEST SET --> CHEATING :D
        model.eval()
        for split_name, test_loader in zip(['pub target', 'priv target'], [test_pub_loader, test_priv_loader]):
            if test_loader is None:
                continue
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(test_loader):
                Xt, yt = Xt.float().to(device), yt.long().to(device)
                with torch.no_grad():
                    yt_logits = model.inference(Xt)
                loss = F.cross_entropy(yt_logits, yt)
                yt_preds = torch.argmax(yt_logits, dim=1)
                acc = torch.mean((yt_preds == yt).float())
                loss_hist.append(loss.item())
                acc_hist.append(acc.item())
            loss = sum(loss_hist)/len(loss_hist)
            acc = sum(acc_hist)/len(acc_hist)
            if cfg['use_visdom']:
                loss_plt.plot('class loss', cfg['target']+' '+split_name, 'XEN', epoch, loss)
                loss_plt.plot('accuracy', cfg['target']+' '+split_name, 'ACC', epoch, acc)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

    return model
