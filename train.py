import argparse
import random
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import np_transforms as NP_T
import plotter
from datasets import Trancos, MergeLoaders
from models import FCN_rLSTM
from utils import show_images


def main():
    parser = argparse.ArgumentParser(description='Train FCN in Trancos dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', default='./fcn.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/TRANCOS_v3', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target_cam', default=1, type=int, metavar='', help='target camera ID')
    parser.add_argument('--valid', default=0.2, type=float, metavar='', help='fraction of the training data for validation')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=500, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, metavar='', help='batch size')
    parser.add_argument('--size_red', default=4, type=int, metavar='', help='size reduction factor to be applied to the images')
    parser.add_argument('--lambda', default=1e-4, type=float, metavar='', help='trade-off between density estimation and vehicle count losses (see eq. 7 in the paper)')
    parser.add_argument('--gamma', default=1e3, type=float, metavar='', help='precision parameter of the Gaussian kernel (inverse of variance)')
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='weight decay regularization')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='FCN_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--n2show', default=8, type=int, metavar='', help='number of examples to show in Visdom in each epoch')
    parser.add_argument('--vis_shape', nargs=2, default=[120, 160], type=int, metavar='', help='shape of the images shown in Visdom')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())

    # dump args to a txt file for your records
    with open(args['model_path'] + '.txt', 'w') as f:
        f.write(str(args)+'\n')

    # use a fixed random seed for reproducibility purposes
    if args['seed'] > 0:
        random.seed(args['seed'])
        np.random.seed(seed=args['seed'])
        torch.manual_seed(args['seed'])

    # if args['use_cuda'] == True and we have a GPU, use the GPU; otherwise, use the CPU
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'
    print('device:', device)

    # define image transformations to be applied to each image in the dataset
    train_transf = T.Compose([
        NP_T.RandomHorizontalFlip(0.5),  # data augmentation: horizontal flipping (we could add more transformations)
        NP_T.ToTensor()  # convert np.array to tensor
    ])
    valid_transf = NP_T.ToTensor()  # no data augmentation in validation

    # instantiate the dataset
    source_cams = list(range(14)).remove(args['target_cam'])
    train_src_data = Trancos(train=True, path=args['data_path'], size_red=args['size_red'],
        transform=train_transf, gamma=args['gamma'], get_cameras=True, cameras=source_cams)
    train_tgt_data = Trancos(train=True, path=args['data_path'], size_red=args['size_red'],
        transform=train_transf, gamma=args['gamma'], get_cameras=True, cameras=[args['target_cam']])
    valid_src_data = Trancos(train=True, path=args['data_path'], size_red=args['size_red'],
        transform=valid_transf, gamma=args['gamma'], get_cameras=True, cameras=source_cams)
    valid_tgt_data = Trancos(train=True, path=args['data_path'], size_red=args['size_red'],
        transform=valid_transf, gamma=args['gamma'], get_cameras=True, cameras=[args['target_cam']])

    # split the data into training and validation sets
    if args['valid'] > 0:
        valid_indices = set(random.sample(range(len(train_src_data)), int(len(train_src_data)*args['valid'])))  # randomly choose some images for validation
        valid_src_data = Subset(valid_src_data, list(valid_indices))
        train_indices = set(range(len(train_src_data))) - valid_indices  # remaining images are for training
        train_src_data = Subset(train_src_data, list(train_indices))

        valid_indices = set(random.sample(range(len(train_tgt_data)), int(len(train_tgt_data)*args['valid'])))  # randomly choose some images for validation
        valid_tgt_data = Subset(valid_tgt_data, list(valid_indices))
        train_indices = set(range(len(train_tgt_data))) - valid_indices  # remaining images are for training
        train_tgt_data = Subset(train_tgt_data, list(train_indices))
    else:
        valid_src_data, valid_tgt_data = None, None

    # create data loaders for training and validation
    train_src_loader = DataLoader(train_src_data,
                                  batch_size=args['batch_size'],
                                  shuffle=True)  # shuffle the data at the beginning of each epoch
    train_tgt_loader = DataLoader(train_tgt_data,
                                  batch_size=args['batch_size'],
                                  shuffle=True)  # shuffle the data at the beginning of each epoch
    train_loader = MergeLoaders(train_src_loader, train_tgt_loader)

    if valid_src_data:
        valid_src_loader = DataLoader(valid_src_data,
                                  batch_size=args['batch_size'],
                                  shuffle=False)  # no need to shuffle in validation
        valid_tgt_loader = DataLoader(valid_tgt_data,
                                  batch_size=args['batch_size'],
                                  shuffle=False)  # no need to shuffle in validation
        valid_loader = MergeLoaders(valid_src_loader, valid_tgt_loader)
    else:
        valid_loader = None

    # instantiate the model and define an optimizer
    model = FCN_rLSTM(temporal=False).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Visdom is a tool to visualize plots during training
    if args['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=args['visdom_env'],
                                             port=args['visdom_port'])
        img_plt = plotter.VisdomImgsPlotter(env_name=args['visdom_env'],
                                            port=args['visdom_port'])

    # training routine
    for epoch in range(args['epochs']):
        print('Epoch {}/{}'.format(epoch, args['epochs']-1))

        # training phase
        model.train()  # set model to training mode (affects batchnorm and dropout, if present)
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        Xsrc, mask_src, density_src, count_src = None, None, None, None
        t0 = time.time()
        for i, (Dsrc, Dtgt) in enumerate(train_loader):
            # unpack source and target data
            Xsrc, mask_src, density_src, count_src, _ = Dsrc
            Xtgt, mask_tgt, _, _, _ = Dtgt

            # copy the tensors to GPU (if applicable)
            Xsrc, mask_src, density_src, count_src = Xsrc.to(device), mask_src.to(device), density_src.to(device), count_src.to(device)

            # forward pass through the model
            density_pred_src, count_pred_src = model(Xsrc, mask=mask_src)
            with torch.no_grad():
                density_pred_tgt, count_pred_tgt = model(Xtgt, mask=mask_tgt)

            # compute the loss
            N = Xsrc.shape[0]
            density_loss = torch.sum((density_pred_src - density_src)**2)/(2*N)
            count_loss = torch.sum((count_pred_src - count_src)**2)/(2*N)
            loss = density_loss + args['lambda']*count_loss

            # backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('{}/{} mini-batch loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f}'
                  .format(i, len(train_loader)-1, loss.item(), density_loss.item(), count_loss.item()),
                  flush=True, end='\r')

            # save the loss values
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            with torch.no_grad():  # evaluation metric, so no need to compute gradients
                count_err = torch.sum(torch.abs(count_pred_src - count_src))/N
            count_err_hist.append(count_err.item())
        t1 = time.time()
        print()

        # print the average training losses
        train_loss = sum(loss_hist)/len(loss_hist)
        train_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        train_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        train_count_err = sum(count_err_hist)/len(count_err_hist)
        print('Training statistics:')
        print('global loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'
              .format(train_loss, train_density_loss, train_count_loss, train_count_err))
        print('time: {:.0f} seconds'.format(t1-t0))

        if args['use_visdom']:
            # plot the losses
            loss_plt.plot('global loss', 'train', 'MSE', epoch, train_loss)
            loss_plt.plot('density loss', 'train', 'MSE', epoch, train_density_loss)
            loss_plt.plot('count loss', 'train', 'MSE', epoch, train_count_loss)
            loss_plt.plot('count error', 'train', 'MAE', epoch, train_count_err)

            # show a few training examples (images + density maps) from source domain
            Xsrc *= mask_src  # show the active region only
            X, density, count = Xsrc.cpu().numpy(), density_src.cpu().numpy(), count_src.cpu().numpy()
            density_pred, count_pred = density_pred.detach().cpu().numpy(), count_pred.detach().cpu().numpy()
            n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
            show_images(img_plt, 'train gt', X[0:n2show], density[0:n2show], count[0:n2show], shape=args['vis_shape'])
            show_images(img_plt, 'train pred', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['vis_shape'])

            # show a few training examples (images + density maps) from target domain
            Xtgt *= mask_tgt  # show the active region only
            X, density_pred, count_pred = Xtgt.cpu().numpy(), density_pred_tgt.cpu().numpy(), count_pred_tgt.cpu().numpy()
            n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
            show_images(img_plt, 'train tgt pred', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['vis_shape'])

        if valid_loader is None:
            print()
            continue

        # validation phase
        model.eval()  # set model to evaluation mode (affects batchnorm and dropout, if present)
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        Xsrc, mask_src, density_src, count_src = None, None, None, None
        Xtgt, mask_tgt = None, None
        t0 = time.time()
        for i, (Dsrc, Dtgt) in enumerate(valid_loader):
            # unpack source and target data
            Xsrc, mask_src, density_src, count_src, _ = Dsrc
            Xtgt, mask_tgt, _, _, _ = Dtgt

            # copy the tensors to GPU (if applicable)
            Xsrc, mask_src, density_src, count_src = Xsrc.to(device), mask_src.to(device), density_src.to(device), count_src.to(device)
            Xtgt, mask_tgt = Xtgt.to(device), mask_tgt.to(device)

            # forward pass through the model
            with torch.no_grad():  # no need to compute gradients in validation (faster and uses less memory)
                density_pred_src, count_pred_src = model(Xsrc, mask=mask_src)
                density_pred_tgt, count_pred_tgt = model(Xtgt, mask=mask_tgt)

            # compute the loss
            N = Xsrc.shape[0]
            density_loss = torch.sum((density_pred_src - density_src)**2)/(2*N)
            count_loss = torch.sum((count_pred_src - count_src)**2)/(2*N)
            loss = density_loss + args['lambda']*count_loss

            # save the loss values
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            count_err = torch.sum(torch.abs(count_pred_src - count_src))/N
            count_err_hist.append(count_err.item())
        t1 = time.time()

        # print the average validation losses
        valid_loss = sum(loss_hist)/len(loss_hist)
        valid_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        valid_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        valid_count_err = sum(count_err_hist)/len(count_err_hist)
        print('Validation statistics:')
        print('global loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'
              .format(valid_loss, valid_density_loss, valid_count_loss, valid_count_err))
        print('time: {:.0f} seconds'.format(t1-t0))
        print()

        if args['use_visdom']:
            # plot the losses
            loss_plt.plot('global loss', 'valid', 'MSE', epoch, valid_loss)
            loss_plt.plot('density loss', 'valid', 'MSE', epoch, valid_density_loss)
            loss_plt.plot('count loss', 'valid', 'MSE', epoch, valid_count_loss)
            loss_plt.plot('count error', 'valid', 'MAE', epoch, valid_count_err)

            # show a few validation examples (images + density maps) from source domain
            Xsrc *= mask_src  # show the active region only
            X, density, count = Xsrc.cpu().numpy(), density_src.cpu().numpy(), count_src.cpu().numpy()
            density_pred, count_pred = density_pred.cpu().numpy(), count_pred.cpu().numpy()
            n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
            show_images(img_plt, 'valid src gt', X[0:n2show], density[0:n2show], count[0:n2show], shape=args['vis_shape'])
            show_images(img_plt, 'valid src pred', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['vis_shape'])

            # show a few validation examples (images + density maps) from target domain
            Xtgt *= mask_tgt  # show the active region only
            X, density_pred, count_pred = Xtgt.cpu().numpy(), density_pred_tgt.cpu().numpy(), count_pred_tgt.cpu().numpy()
            n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
            show_images(img_plt, 'valid tgt pred', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['vis_shape'])

    torch.save(model.state_dict(), args['model_path'])


if __name__ == '__main__':
    main()
