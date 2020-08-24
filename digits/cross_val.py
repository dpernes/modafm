import sys
sys.path.append('..')

import argparse
from configparser import ConfigParser
import os
import random
import ast

import numpy as np
import torch
from torch.utils.data import Subset
import torchvision.transforms as T

from datasets import MNIST, MNIST_M, SVHN, SynthDigits
from models import MDANet, MODANet
from routines import (cross_validation, mdan_train_routine,
                      moda_train_routine, moda_fm_train_routine)
from utils import Logger
from augment import Flip


def main():
    # N.B.: parameters defined in cv_cfg.ini override args!
    parser = argparse.ArgumentParser(description='Cross-validation over source domains for the digits datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MODAFM', type=str, metavar='', help='model type (\'MDAN\' / \'MODA\' / \'MODAFM\'')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='MNIST', type=str, metavar='', help='target domain (\'MNIST\' / \'MNIST_M\' / \'SVHN\' / \'SynthDigits\')')
    parser.add_argument('-o', '--output', default='msda_hyperparams.ini', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('-n', '--n_iter', default=20, type=int, metavar='', help='number of CV iterations')
    parser.add_argument('--n_images', default=20000, type=int, metavar='', help='number of images from each domain')
    parser.add_argument('--mu', type=float, default=1e-2, help="hyperparameter of the coefficient for the domain adversarial loss")
    parser.add_argument('--beta', type=float, default=0.2, help="hyperparameter of the non-sparsity regularization")
    parser.add_argument('--lambda', type=float, default=1e-1, help="hyperparameter of the FixMatch loss")
    parser.add_argument('--n_rand_aug', type=int, default=2, help="N parameter of RandAugment")
    parser.add_argument('--m_min_rand_aug', type=int, default=3, help="minimum M parameter of RandAugment")
    parser.add_argument('--m_max_rand_aug', type=int, default=10, help="maximum M parameter of RandAugment")
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='hyperparameter of weight decay regularization')
    parser.add_argument('--lr', default=1e-1, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=30, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, metavar='', help='batch size (per domain)')
    parser.add_argument('--checkpoint', default=0, type=int, metavar='', help='number of epochs between saving checkpoints (0 disables checkpoints)')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='digits_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--verbosity', default=2, type=int, metavar='', help='log verbosity level (0, 1, 2)')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())

    # override args with cv_cfg.ini
    cfg = args.copy()
    cv_parser = ConfigParser()
    cv_parser.read('cv_cfg.ini')
    cv_param_names = []
    for key, val in cv_parser.items('main'):
        cfg[key] = ast.literal_eval(val)
        cv_param_names.append(key)

    # use a fixed random seed for reproducibility purposes
    if cfg['seed'] > 0:
        random.seed(cfg['seed'])
        np.random.seed(seed=cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed(cfg['seed'])

    device = 'cuda' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    log = Logger(cfg['verbosity'])
    log.print('device:', device, level=0)

    if 'FM' in cfg['model']:
        # weak data augmentation (small rotation + small translation)
        data_aug = T.Compose([
            T.RandomAffine(5, translate=(0.125, 0.125)),
            T.ToTensor(),
        ])
    else:
        data_aug = T.ToTensor()
    cfg['test_transform'] = T.ToTensor()

    # define all datasets
    datasets = {}
    datasets['MNIST'] = MNIST(train=True, path=os.path.join(cfg['data_path'], 'MNIST'), transform=data_aug)
    datasets['MNIST_M'] = MNIST_M(train=True, path=os.path.join(cfg['data_path'], 'MNIST_M'), transform=data_aug)
    datasets['SVHN'] = SVHN(train=True, path=os.path.join(cfg['data_path'], 'SVHN'), transform=data_aug)
    datasets['SynthDigits'] = SynthDigits(train=True, path=os.path.join(cfg['data_path'], 'SynthDigits'), transform=data_aug)
    del datasets[cfg['target']]

    # get a subset of cfg['n_images'] from each dataset
    for ds_name in datasets:
        if ds_name == cfg['target']:
            continue
        indices = random.sample(range(len(datasets[ds_name])), cfg['n_images'])
        datasets[ds_name] = Subset(datasets[ds_name], indices[0:cfg['n_images']])

    if cfg['model'] == 'MDAN':
        cfg['model'] = MDANet(len(datasets)-1).to(device)
        cfg['train_routine'] = lambda model, optimizer, train_loader, cfg: mdan_train_routine(model, optimizer, train_loader, dict(), cfg)
    elif cfg['model'] == 'MODA':
        cfg['model'] = MODANet().to(device)
        cfg['train_routine'] = lambda model, optimizer, train_loader, cfg: moda_train_routine(model, optimizer, train_loader, dict(), cfg)
    elif cfg['model'] == 'MODAFM':
        cfg['model'] = MODANet().to(device)
        cfg['excl_transf'] = [Flip]
        cfg['train_routine'] = lambda model, optimizer, train_loader, cfg: moda_fm_train_routine(model, optimizer, train_loader, dict(), cfg)
    else:
        raise ValueError('Unknown model {}'.format(cfg['model']))

    best_params, _ = cross_validation(datasets, cfg, cv_param_names)
    log.print('best_params:', best_params, level=1)

    results = ConfigParser()
    results.add_section('main')
    for key, value in  best_params.items():
        results.set('main', key, str(value))
    with open(cfg['output'], 'w') as f:
        results.write(f)

if __name__ == '__main__':
    main()
