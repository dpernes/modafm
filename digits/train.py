import sys
sys.path.append('..')

import argparse
from configparser import ConfigParser
import ast
import os
import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T

from datasets import MNIST, MNIST_M, SVHN, SynthDigits
from models import SimpleCNN, MDANet, MixMDANet
from routines import (fs_train_routine, fm_train_routine, dann_train_routine, mdan_train_routine, mdan_unif_train_routine,
                      mdan_fm_train_routine, mdan_unif_fm_train_routine, mixmdan_train_routine, mixmdan_fm_train_routine)
from utils import MSDA_Loader, Logger
from augment import Flip


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with digits datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MDAN', type=str, metavar='', help='model type (\'FS\' / \'MDAN\' / \'MixMDAN\' / \'FM\' / \'MixMDANFM\')')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='MNIST', type=str, metavar='', help='target domain (\'MNIST\' / \'MNIST_M\' / \'SVHN\' / \'SynthDigits\')')
    parser.add_argument('-o', '--output', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--icfg', default=None, type=str, metavar='', help='config file (overrides args)')
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
    parser.add_argument('--eval_target', default=False, type=int, metavar='', help='evaluate target during training')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='digits_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--verbosity', default=2, type=int, metavar='', help='log verbosity level (0, 1, 2)')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())

    # override args with icfg (if provided)
    cfg = args.copy()
    if cfg['icfg'] is not None:
        cv_parser = ConfigParser()
        cv_parser.read(cfg['icfg'])
        cv_param_names = []
        for key, val in cv_parser.items('main'):
            cfg[key] = ast.literal_eval(val)
            cv_param_names.append(key)

    # dump cfg to a txt file for your records
    with open(cfg['output'] + '.txt', 'w') as f:
        f.write(str(cfg)+'\n')

    # use a fixed random seed for reproducibility purposes
    if cfg['seed'] > 0:
        random.seed(cfg['seed'])
        np.random.seed(seed=cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed(cfg['seed'])

    device = 'cuda' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    log = Logger(cfg['verbosity'])
    log.print('device:', device, level=0)

    if ('FS' in cfg['model']) or ('FM' in cfg['model']):
        # weak data augmentation (small rotation + small translation)
        data_aug = T.Compose([
            T.RandomAffine(5, translate=(0.125, 0.125)),
            T.ToTensor(),
        ])
    else:
        data_aug = T.ToTensor()

    # define all datasets
    datasets = {}
    datasets['MNIST'] = MNIST(train=True, path=os.path.join(cfg['data_path'], 'MNIST'), transform=data_aug)
    datasets['MNIST_M'] = MNIST_M(train=True, path=os.path.join(cfg['data_path'], 'MNIST_M'), transform=data_aug)
    datasets['SVHN'] = SVHN(train=True, path=os.path.join(cfg['data_path'], 'SVHN'), transform=data_aug)
    datasets['SynthDigits'] = SynthDigits(train=True, path=os.path.join(cfg['data_path'], 'SynthDigits'), transform=data_aug)
    if ('FS' in cfg['model']) or ('FM' in cfg['model']):
        test_set = deepcopy(datasets[cfg['target']])
        test_set.transform = T.ToTensor()  # no data augmentation in test
    else:
        test_set = datasets[cfg['target']]

    # get a subset of cfg['n_images'] from each dataset
    # define public and private test sets: the private is not used at training time to learn invariant representations
    for ds_name in datasets:
        if ds_name == cfg['target']:
            indices = random.sample(range(len(datasets[ds_name])), 2*cfg['n_images'])
            test_pub_set = Subset(test_set, indices[0:cfg['n_images']])
            test_priv_set = Subset(test_set, indices[cfg['n_images']::])
        else:
            indices = random.sample(range(len(datasets[ds_name])), cfg['n_images'])
        datasets[ds_name] = Subset(datasets[ds_name], indices[0:cfg['n_images']])

    # build the dataloader
    train_loader = MSDA_Loader(datasets, cfg['target'], batch_size=cfg['batch_size'], shuffle=True, device=device)
    test_pub_loader = DataLoader(test_pub_set, batch_size=4*cfg['batch_size'])
    test_priv_loader = DataLoader(test_priv_set, batch_size=4*cfg['batch_size'])
    valid_loaders = ({'target pub': test_pub_loader, 'target priv': test_priv_loader}
                     if cfg['eval_target'] else None)
    log.print('target domain:', cfg['target'], '| source domains:', train_loader.sources, level=1)

    if cfg['model'] == 'FS':
        model = SimpleCNN().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        if valid_loaders is not None:
            del valid_loaders['target pub']
        fs_train_routine(model, optimizer, test_pub_loader, valid_loaders, cfg)
    elif cfg['model'] == 'FM':
        model = SimpleCNN().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        cfg['excl_transf'] = [Flip]
        fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'DANNS':
        for src in train_loader.sources:
            model = MixMDANet().to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            dataset_ss = {src: datasets[src], cfg['target']: datasets[cfg['target']]}
            train_loader = MSDA_Loader(dataset_ss, cfg['target'], batch_size=cfg['batch_size'], shuffle=True, device=device)
            dann_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
            torch.save(model.state_dict(), cfg['output']+'_'+src)
    elif cfg['model'] == 'DANNM':
        model = MixMDANet().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        dann_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MDAN':
        model = MDANet(len(train_loader.sources)).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        mdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MDANU':
        model = MDANet(len(train_loader.sources)).to(device)
        model.grad_reverse = nn.ModuleList([nn.Identity() for _ in range(len(model.domain_class))])  # remove grad reverse
        task_optim = optim.Adadelta(list(model.feat_ext.parameters())+list(model.task_class.parameters()),
                                    lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        adv_optim = optim.Adadelta(model.domain_class.parameters(),
                                   lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        optimizers = (task_optim, adv_optim)
        mdan_unif_train_routine(model, optimizers, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MDANFM':
        model = MDANet(len(train_loader.sources)).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        mdan_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MDANUFM':
        model = MDANet(len(train_loader.sources)).to(device)
        task_optim = optim.Adadelta(list(model.feat_ext.parameters())+list(model.task_class.parameters()),
                                    lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        adv_optim = optim.Adadelta(model.domain_class.parameters(),
                                   lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        optimizers = (task_optim, adv_optim)
        cfg['excl_transf'] = [Flip]
        mdan_unif_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MixMDAN':
        model = MixMDANet().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        mixmdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MixMDANFM':
        model = MixMDANet().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        cfg['excl_transf'] = [Flip]
        mixmdan_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    else:
        raise ValueError('Unknown model {}'.format(cfg['model']))

    if cfg['model'] != 'DANNS':
        torch.save(model.state_dict(), cfg['output'])

if __name__ == '__main__':
    main()
