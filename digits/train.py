import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T

from datasets import MNIST, MNIST_M, SVHN, SynthDigits
from routines import (mdan_train_routine, mdan_unif_train_routine, mdan_fm_train_routine,
                      mdan_unif_fm_train_routine, mixmdan_train_routine, mixmdan_fm_train_routine)
from utils import MSDA_Loader
import plotter

def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with digits datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MDAN', type=str, metavar='', help='model type (\'MDAN\' / \'MDANU\' / \'MDANFM\' / \'MDANUFM\' / \'MixMDAN\' / \'MixMDANFM\')')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='MNIST', type=str, metavar='', help='target domain (\'MNIST\' / \'MNIST_M\' / \'SVHN\' / \'SynthDigits\')')
    parser.add_argument('-o', '--output', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--mode', default='dynamic', type=str, metavar='', help='mode of combination rule (\'dynamic\' / \'minmax\')')
    parser.add_argument('--n_images', default=20000, type=int, metavar='', help='number of images from each domain')
    parser.add_argument('--mu', type=float, default=1e-2, help="hyperparameter of the coefficient for the domain adversarial loss")
    parser.add_argument('--gamma', type=float, default=10., help="hyperparameter of the dynamic loss")
    parser.add_argument('--beta', type=float, default=0.2, help="hyperparameter of the non-sparsity regularization")
    parser.add_argument('--lambda', type=float, default=1e-1, help="hyperparameter of the FixMatch loss")
    parser.add_argument('--n_rand_aug', type=int, default=2, help="N parameter of RandAugment")
    parser.add_argument('--m_min_rand_aug', type=int, default=3, help="minimum M parameter of RandAugment")
    parser.add_argument('--m_max_rand_aug', type=int, default=10, help="maximum M parameter of RandAugment")
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='hyperparameter of weight decay regularization')
    parser.add_argument('--lr', default=1e-2, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=30, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, metavar='', help='batch size (per domain)')
    parser.add_argument('--checkpoint', default=0, type=int, metavar='', help='number of epochs between saving checkpoints (0 disables checkpoints)')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='digits_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())

    # dump args to a txt file for your records
    with open(args['output'] + '.txt', 'w') as f:
        f.write(str(args)+'\n')

    # use a fixed random seed for reproducibility purposes
    if args['seed'] > 0:
        random.seed(args['seed'])
        np.random.seed(seed=args['seed'])
        torch.manual_seed(args['seed'])

    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    print('device:', device)

    if 'FM' in args['model']:
        # weak data augmentation (small rotation + small translation)
        data_aug = T.Compose([
            T.RandomAffine(5, translate=(0.125, 0.125)),
            T.ToTensor(),
        ])
    else:
        data_aug = T.ToTensor()

    # define all datasets
    datasets = {}
    datasets['MNIST'] = MNIST(train=True, path=os.path.join(args['data_path'], 'MNIST'), transform=data_aug)
    datasets['MNIST_M'] = MNIST_M(train=True, path=os.path.join(args['data_path'], 'MNIST_M'), transform=data_aug)
    datasets['SVHN'] = SVHN(train=True, path=os.path.join(args['data_path'], 'SVHN'), transform=data_aug)
    datasets['SynthDigits'] = SynthDigits(train=True, path=os.path.join(args['data_path'], 'SynthDigits'), transform=data_aug)
    if 'FM' in args['model']:
        test_set = deepcopy(datasets[args['target']])
        test_set.transform = T.ToTensor()  # no data augmentation in test
    else:
        test_set = datasets[args['target']]

    # get a subset of args['n_images'] from each dataset
    # define public and private test sets: the private is not used at training time to learn invariant representations
    for ds_name in datasets:
        if ds_name == args['target']:
            indices = random.sample(range(len(datasets[ds_name])), 2*args['n_images'])
            test_pub_set = Subset(test_set, indices[0:args['n_images']])
            test_priv_set = Subset(test_set, indices[args['n_images']::])
        else:
            indices = random.sample(range(len(datasets[ds_name])), args['n_images'])
        datasets[ds_name] = Subset(datasets[ds_name], indices[0:args['n_images']])

    # build the dataloader
    train_loader = MSDA_Loader(datasets, args['target'], batch_size=args['batch_size'], shuffle=True, device=device)
    test_pub_loader = DataLoader(test_pub_set, batch_size=4*args['batch_size'])
    test_priv_loader = DataLoader(test_priv_set, batch_size=4*args['batch_size'])
    print('source domains:', train_loader.sources)

    if args['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=args['visdom_env'], port=args['visdom_port'])
    else:
        loss_plt = None

    if args['model'] == 'MDAN':
        model = mdan_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, args)
    elif args['model'] == 'MDANU':
        model = mdan_unif_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, args)
    elif args['model'] == 'MDANFM':
        model = mdan_fm_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, args)
    elif args['model'] == 'MDANUFM':
        model = mdan_unif_fm_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, args)
    elif args['model'] == 'MixMDAN':
        model = mixmdan_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, args)
    elif args['model'] == 'MixMDANFM':
        model = mixmdan_fm_train_routine(train_loader, test_pub_loader, test_priv_loader, loss_plt, args)
    else:
        raise ValueError('Unknown model {}'.format(args['model']))

    torch.save(model.state_dict(), args['output'])

if __name__ == '__main__':
    main()
