import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gradient_reversal import GradientReversalLayer
from augment import RandAugment
from utils import (Logger, fixmatch_loss, save_checkpoint,
                   loguniform, reset_model, eval_accuracy)
import plotter


def mdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    if cfg['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=cfg['visdom_env'], port=cfg['visdom_port'])
    log = Logger(cfg['verbosity'])

    for epoch in range(cfg['epochs']):
        log.print('Epoch {}/{}'.format(epoch, cfg['epochs']-1), level=1)
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
            loss = torch.log(torch.sum(torch.exp(10*(class_losses + cfg['mu']*domain_losses))))/10
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            log.print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                      flush=True, end='\r', level=0)
            loss_hist.append(loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                domain_losses_hist[j].append(domain_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        log.print(level=0)
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_losses = [sum(domain_losses_hist[i])/len(domain_losses_hist[i]) for i in range(len(train_loader.sources))]
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        log.print('global loss: {:.3f}'.format(loss)
                  +' | class losses: ' + str([float('{:.3f}'.format(cl)) for cl in class_losses])
                  +' | domain losses: ' + str([float('{:.3f}'.format(dl)) for dl in domain_losses]),
                  level=1)
        log.print('accuracies: '+ str([float('{:.3f}'.format(acc)) for acc in accuracies]), level=1)
        log.print('{:.1f} seconds'.format(t1-t0), level=1)
        log.print(level=1)
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', 'total', epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

        if valid_loaders is None:
            continue

        # EVALUATE
        model.eval()
        for split_name, loader in valid_loaders.items():
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(loader):
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


def mdan_unif_train_routine(model, optimizers, train_loader, valid_loaders, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    task_optim, adv_optim = optimizers
    if cfg['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=cfg['visdom_env'], port=cfg['visdom_port'])
    log = Logger(cfg['verbosity'])

    for epoch in range(cfg['epochs']):
        log.print('Epoch {}/{}'.format(epoch, cfg['epochs']-1), level=1)
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
            loss = torch.log(torch.sum(torch.exp(10*(class_losses + cfg['mu']*unif_losses))))/10
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
            log.print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                      flush=True, end='\r', level=0)
            loss_hist.append(loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                domain_losses_hist[j].append(domain_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        log.print(level=0)
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_losses = [sum(domain_losses_hist[i])/len(domain_losses_hist[i]) for i in range(len(train_loader.sources))]
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        log.print('global loss: {:.3f}'.format(loss)
                  +' | class losses: ' + str([float('{:.3f}'.format(cl)) for cl in class_losses])
                  +' | domain losses: ' + str([float('{:.3f}'.format(dl)) for dl in domain_losses]),
                  level=1)
        log.print('accuracies: '+ str([float('{:.3f}'.format(acc)) for acc in accuracies]), level=1)
        log.print('{:.1f} seconds'.format(t1-t0), level=1)
        log.print(level=1)
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', 'total', epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        if valid_loaders is None:
            continue

        # EVALUATE
        model.eval()
        for split_name, loader in valid_loaders.items():
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(loader):
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


def mdan_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    if cfg['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=cfg['visdom_env'], port=cfg['visdom_port'])
    log = Logger(cfg['verbosity'])

    for epoch in range(cfg['epochs']):
        log.print('Epoch {}/{}'.format(epoch, cfg['epochs']-1), level=1)
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
            aug_transf = lambda batch: torch.stack([
                RandAugment(cfg['n_rand_aug'], m_aug, cutout=int(0.3*Xt.shape[2]), exclusions=cfg['excl_transf'])(img)
                for img in batch])
            Xt_aug = aug_transf(Xt.cpu()).to(device)

            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            yt_logits = model.inference(Xt)
            yt_aug_logits = model.inference(Xt_aug)

            class_losses = torch.stack([F.cross_entropy(ys_logits[i], ys[i]) for i in range(len(train_loader.sources))])
            domain_losses = torch.stack([F.cross_entropy(ds_logits[i], ds[i]) + F.cross_entropy(dt_logits[i], dt)
                                         for i in range(len(train_loader.sources))])
            fm_loss = fixmatch_loss(yt_logits, yt_aug_logits)
            if 'total' == 'dynamic':
                loss = torch.log(torch.sum(torch.exp(10*(class_losses + cfg['mu']*domain_losses))))/10
            else:
                loss = torch.max(class_losses + cfg['mu']*domain_losses)
            loss += cfg['lambda']*fm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ys_preds = [torch.argmax(ys_logits[j], dim=1) for j in range(len(train_loader.sources))]
            acc = [torch.mean((ys_preds[j] == ys[j]).float()) for j in range(len(train_loader.sources))]
            log.print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                      flush=True, end='\r', level=0)
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
        log.print('global loss: {:.3f}'.format(loss)
                  +' | class losses: ' + str([float('{:.3f}'.format(cl)) for cl in class_losses])
                  +' | domain losses: ' + str([float('{:.3f}'.format(dl)) for dl in domain_losses]),
                  level=1)
        log.print('accuracies: '+ str([float('{:.3f}'.format(acc)) for acc in accuracies]), level=1)
        log.print('{:.1f} seconds'.format(t1-t0), level=1)
        log.print(level=1)
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', 'total', epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

        if valid_loaders is None:
            continue

        # EVALUATE
        model.eval()
        for split_name, loader in valid_loaders.items():
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(loader):
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


def mdan_unif_fm_train_routine(model, optimizers, train_loader, valid_loaders, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    task_optim, adv_optim = optimizers
    if cfg['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=cfg['visdom_env'], port=cfg['visdom_port'])
    log = Logger(cfg['verbosity'])

    for epoch in range(cfg['epochs']):
        log.print('Epoch {}/{}'.format(epoch, cfg['epochs']-1), level=1)
        model.train()
        loss_hist, fm_loss_hist = [], []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_losses_hist = [[] for i in range(len(train_loader.sources))]
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)

            m_aug = np.random.randint(cfg['m_min_rand_aug'], cfg['m_max_rand_aug']+1)
            aug_transf = lambda batch: torch.stack([
                RandAugment(cfg['n_rand_aug'], m_aug, cutout=int(0.3*Xt.shape[2]), exclusions=cfg['excl_transf'])(img)
                for img in batch])
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
            if 'total' == 'dynamic':
                loss = torch.log(torch.sum(torch.exp(10*(class_losses + cfg['mu']*unif_losses))))/10
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
            log.print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                      flush=True, end='\r', level=0)
            loss_hist.append(loss.item())
            fm_loss_hist.append(fm_loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                domain_losses_hist[j].append(domain_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        log.print()
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_losses = [sum(domain_losses_hist[i])/len(domain_losses_hist[i]) for i in range(len(train_loader.sources))]
        fm_loss = sum(fm_loss_hist)/len(fm_loss_hist)
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        print('global loss: {:.3f}'.format(loss)
              +' | class losses: ' + str([float('{:.3f}'.format(cl)) for cl in class_losses])
              +' | domain losses: ' + str([float('{:.3f}'.format(dl)) for dl in domain_losses])
              +' | fixmatch loss: {:.3f}'.format(fm_loss),
              level=1)
        log.print('accuracies: '+ str([float('{:.3f}'.format(acc)) for acc in accuracies]), level=1)
        log.print('{:.1f} seconds'.format(t1-t0), level=1)
        log.print(level=1)
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', 'total', epoch, loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('domain loss', train_loader.sources[i], 'XEN', epoch, domain_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])
            loss_plt.plot('fixmatch loss', 'train', 'XEN', epoch, fm_loss)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, [task_optim, adv_optim], None, name=cfg['output']+'.ckp'+str(epoch))

        if valid_loaders is None:
            continue

        # EVALUATE
        model.eval()
        for split_name, loader in valid_loaders.items():
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(loader):
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


def mixmdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    alpha = nn.Parameter(torch.Tensor(len(train_loader.sources)).to(device))
    nn.init.uniform_(alpha)
    grad_reverse_fn = GradientReversalLayer().to(device)
    optimizer.add_param_group({'params': alpha, 'lr': cfg['lr']})
    if cfg['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=cfg['visdom_env'], port=cfg['visdom_port'])
    log = Logger(cfg['verbosity'])

    for epoch in range(cfg['epochs']):
        log.print('Epoch {}/{}'.format(epoch, cfg['epochs']-1), level=1)
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
            log.print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                      flush=True, end='\r', level=0)
            loss_hist.append(loss.item())
            domain_loss_hist.append(mix_domain_loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        log.print(level=0)
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_loss = sum(domain_loss_hist)/len(domain_loss_hist)
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        log.print('global loss: {:.3f}'.format(loss)
                  +' | class losses: ' + str([float('{:.3f}'.format(cl)) for cl in class_losses])
                  +' | domain loss: {:.3f}'.format(domain_loss),
                  level=1)
        log.print('accuracies: '+ str([float('{:.3f}'.format(acc)) for acc in accuracies]), level=1)
        log.print('mix coef: '+ str([float('{:.3f}'.format(coef)) for coef in beta]), level=1)
        log.print('{:.1f} seconds'.format(t1-t0), level=1)
        log.print(level=1)
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', 'total', epoch, loss)
            loss_plt.plot('domain loss', 'train', 'XEN', epoch, domain_loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

        if valid_loaders is None:
            continue

        # EVALUATE
        model.eval()
        for split_name, loader in valid_loaders.items():
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(loader):
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


def mixmdan_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    alpha = nn.Parameter(torch.Tensor(len(train_loader.sources)).to(device))
    nn.init.uniform_(alpha)
    grad_reverse_fn = GradientReversalLayer().to(device)
    optimizer.add_param_group({'params': alpha, 'lr': cfg['lr']})
    if cfg['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=cfg['visdom_env'], port=cfg['visdom_port'])
    log = Logger(cfg['verbosity'])

    for epoch in range(cfg['epochs']):
        log.print('Epoch {}/{}'.format(epoch, cfg['epochs']-1), level=1)
        model.train()
        loss_hist, fm_loss_hist = [], []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_loss_hist = []
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)

            m_aug = np.random.randint(cfg['m_min_rand_aug'], cfg['m_max_rand_aug']+1)
            aug_transf = lambda batch: torch.stack([
                RandAugment(cfg['n_rand_aug'], m_aug, cutout=int(0.3*Xt.shape[2]), exclusions=cfg['excl_transf'])(img)
                for img in batch])
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
            log.print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                      flush=True, end='\r', level=0)
            loss_hist.append(loss.item())
            domain_loss_hist.append(mix_domain_loss.item())
            fm_loss_hist.append(fm_loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        log.print(level=0)
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_loss = sum(domain_loss_hist)/len(domain_loss_hist)
        fm_loss = sum(fm_loss_hist)/len(fm_loss_hist)
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        log.print('global loss: {:.3f}'.format(loss)
                  +' | class losses: ' + str([float('{:.3f}'.format(cl)) for cl in class_losses])
                  +' | domain loss: {:.3f}'.format(domain_loss)
                  +' | fixmatch loss: {:.3f}'.format(fm_loss),
                  level=1)
        log.print('accuracies: '+ str([float('{:.3f}'.format(acc)) for acc in accuracies]), level=1)
        log.print('mix coef: '+ str([float('{:.3f}'.format(coef)) for coef in beta]), level=1)
        log.print('{:.1f} seconds'.format(t1-t0), level=1)
        log.print(level=1)
        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', 'total', epoch, loss)
            loss_plt.plot('domain loss', 'train', 'XEN', epoch, domain_loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])
            loss_plt.plot('fixmatch loss', 'train', 'XEN', epoch, fm_loss)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

        if valid_loaders is None:
            continue

        # EVALUATE
        model.eval()
        for split_name, loader in valid_loaders.items():
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(loader):
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

def mixmdan_mlp_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg):
    device = 'cuda:0' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    alpha = nn.Parameter(torch.Tensor(len(train_loader.sources)).to(device))
    nn.init.uniform_(alpha)
    grad_reverse_fn = GradientReversalLayer().to(device)
    optimizer.add_param_group({'params': alpha, 'lr': cfg['lr']})
    if cfg['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=cfg['visdom_env'], port=cfg['visdom_port'])
    log = Logger(cfg['verbosity'])

    for epoch in range(cfg['epochs']):
        log.print('Epoch {}/{}'.format(epoch, cfg['epochs']-1), level=1)
        model.train()
        loss_hist, fm_loss_hist = [], []
        class_losses_hist = [[] for i in range(len(train_loader.sources))]
        domain_loss_hist = []
        accuracies_hist = [[] for i in range(len(train_loader.sources))]
        t0 = time.time()
        for i, (Xt, Xs, ys) in enumerate(train_loader):
            ds = [torch.zeros(Xs[i].shape[0]).long().to(device)
                  for i in range(len(train_loader.sources))]
            dt = torch.ones(Xt.shape[0]).long().to(device)

            ys_logits, ds_logits, dt_logits = model(Xs, Xt)
            yt_logits = model.inference(Xt)

            dropout_rate = np.random.uniform(low=cfg['min_dropout'], high=cfg['max_dropout'])
            model.set_dropout_rate(dropout_rate)
            yt_aug_logits = model.inference(Xt)
            model.set_dropout_rate(0.)

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
            if cfg['verbosity'] > 1:
                log.print('{}/{} mini-batch loss: {:.3f}'.format(i, len(train_loader)-1, loss.item()),
                          flush=True, end='\r', level=0)
            loss_hist.append(loss.item())
            domain_loss_hist.append(mix_domain_loss.item())
            fm_loss_hist.append(fm_loss.item())
            for j in range(len(train_loader.sources)):
                class_losses_hist[j].append(class_losses[j].item())
                accuracies_hist[j].append(acc[j].item())
        log.print(level=0)
        loss = sum(loss_hist)/len(loss_hist)
        class_losses = [sum(class_losses_hist[i])/len(class_losses_hist[i]) for i in range(len(train_loader.sources))]
        domain_loss = sum(domain_loss_hist)/len(domain_loss_hist)
        fm_loss = sum(fm_loss_hist)/len(fm_loss_hist)
        accuracies = [sum(accuracies_hist[i])/len(accuracies_hist[i]) for i in range(len(train_loader.sources))]
        t1 = time.time()
        log.print('global loss: {:.3f}'.format(loss)
                  +' | class losses: ' + str([float('{:.3f}'.format(cl)) for cl in class_losses])
                  +' | domain loss: {:.3f}'.format(domain_loss)
                  +' | fixmatch loss: {:.3f}'.format(fm_loss),
                  level=1)
        log.print('accuracies: '+ str([float('{:.3f}'.format(acc)) for acc in accuracies]), level=0)
        log.print('mix coef: '+ str([float('{:.3f}'.format(coef)) for coef in beta]), level=1)
        log.print('{:.1f} seconds'.format(t1-t0), level=1)
        log.print(level=1)

        if cfg['use_visdom']:
            loss_plt.plot('global loss', 'train', 'total', epoch, loss)
            loss_plt.plot('domain loss', 'train', 'XEN', epoch, domain_loss)
            for i in range(len(train_loader.sources)):
                loss_plt.plot('class loss', train_loader.sources[i], 'XEN', epoch, class_losses[i])
                loss_plt.plot('accuracy', train_loader.sources[i], 'ACC', epoch, accuracies[i])
            loss_plt.plot('fixmatch loss', 'train', 'XEN', epoch, fm_loss)

        if (cfg['checkpoint'] != 0) and ((epoch+1)%cfg['checkpoint'] == 0):
            save_checkpoint(epoch, model, optimizer, None, name=cfg['output']+'.ckp'+str(epoch))

        if valid_loaders is None:
            continue

        # EVALUATE
        model.eval()
        for split_name, loader in valid_loaders.items():
            loss_hist = []
            acc_hist = []
            for i, (Xt, yt) in enumerate(loader):
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

def cross_validation(datasets, cfg, cv_keys):
    parameters = []
    accuracies = []
    device = 'cuda' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    log = Logger(cfg['verbosity'])
    for i in range(cfg['n_iter']):
        log.print('CV iteration {}/{}'.format(i, cfg['n_iter']-1), level=2)
        cfg_i = cfg.copy()
        cfg_i['verbosity'] -= 2  # decrease the verbosity in the training routine

        # randomly pick a value for each hyperparameter
        param_dict = dict()
        for key in cv_keys:
            if isinstance(cfg[key], list):
                cfg_i[key] = loguniform(cfg[key][0], cfg[key][1])
            elif isinstance(cfg[key], tuple):
                cfg_i[key] = random.choice(cfg[key])
            param_dict[key] = cfg_i[key]
        parameters.append(param_dict)
        log.print('hyperparameters:', param_dict, level=1)

        # train using each source domain as target and evaluate the accuracy
        acc_per_domain = []
        for target in datasets.keys():
            train_loader = MSDA_Loader(datasets, target, batch_size=cfg_i['batch_size'], shuffle=True, device=device)
            reset_model(cfg_i['model'])

            if 'param_groups' in cfg_i:
                for i, group in enumerate(cfg_i['param_groups']):
                    lr = cfg_i['lr'+str(i)] if 'lr'+str(i) in cfg_i else group['lr']
                    weight_decay = cfg_i['weight_decay'+str(i)] if 'weight_decay'+str(i) in cfg_i else group['weight_decay']
                    if i == 0:
                        optimizer = optim.Adadelta([{'params':group['params'], 'lr':lr, 'weight_decay':weight_decay}])
                    else:
                        optimizer.add_param_group({'params':group['params'], 'lr':lr, 'weight_decay':weight_decay})
            else:
                optimizer = optim.Adadelta(cfg_i['model'].parameters(), lr=cfg_i['lr'], weight_decay=cfg_i['weight_decay'])

            cfg_i['train_routine'](cfg_i['model'], optimizer, train_loader, cfg_i)

            target_set = deepcopy(datasets[target])
            target_set.transform = cfg_i['test_transform']
            target_loader = DataLoader(target_set, batch_size=cfg_i['batch_size'])
            acc = eval_accuracy(cfg_i['model'], target_loader, device=device)
            acc_per_domain.append(acc)

        # save the average accuracy across domains
        acc_mean = sum(acc_per_domain)/len(acc_per_domain)
        accuracies.append(acc_mean)
        log.print('average accuracy = {:.3f}'.format(acc_mean), level=2)
        log.print(level=2)

    accuracies = np.array(accuracies)
    i = np.argmax(accuracies)
    return parameters[i], accuracies[i]