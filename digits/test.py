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
from models import SimpleCNN, MDANet, MODANet
from routines import test_routine
from utils import MSDA_Loader, Logger
from augment import Flip


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with digits datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MODAFM', type=str, metavar='', help='model type (\'FS\' / \'DANNS\' / \'DANNM\' / \'MDAN\' / \'MODA\' / \'FM\' / \'MODAFM\'')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='MNIST', type=str, metavar='', help='target domain (\'MNIST\' / \'MNIST_M\' / \'SVHN\' / \'SynthDigits\')')
    parser.add_argument('-i', '--input', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--n_images', default=20000, type=int, metavar='', help='number of images from each domain')
    parser.add_argument('--batch_size', default=8, type=int, metavar='', help='batch size')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--verbosity', default=2, type=int, metavar='', help='log verbosity level (0, 1, 2)')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())
    cfg = args.copy()

    # use a fixed random seed for reproducibility purposes
    if cfg['seed'] > 0:
        random.seed(cfg['seed'])
        np.random.seed(seed=cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed(cfg['seed'])

    device = 'cuda' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    log = Logger(cfg['verbosity'])
    log.print('device:', device, level=0)

    # define all datasets
    datasets = {}
    datasets['MNIST'] = MNIST(train=True, path=os.path.join(cfg['data_path'], 'MNIST'), transform=T.ToTensor())
    datasets['MNIST_M'] = MNIST_M(train=True, path=os.path.join(cfg['data_path'], 'MNIST_M'), transform=T.ToTensor())
    datasets['SVHN'] = SVHN(train=True, path=os.path.join(cfg['data_path'], 'SVHN'), transform=T.ToTensor())
    datasets['SynthDigits'] = SynthDigits(train=True, path=os.path.join(cfg['data_path'], 'SynthDigits'), transform=T.ToTensor())
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
    test_pub_loader = DataLoader(test_pub_set, batch_size=4*cfg['batch_size'])
    test_priv_loader = DataLoader(test_priv_set, batch_size=4*cfg['batch_size'])
    test_loaders = {'target pub': test_pub_loader, 'target priv': test_priv_loader}
    log.print('target domain:', cfg['target'], level=0)

    if cfg['model'] in ['FS', 'FM']:
        model = SimpleCNN().to(device)
    elif args['model'] == 'MDAN':
        model = MDANet(len(datasets)-1).to(device)
    elif cfg['model'] in ['DANNS', 'DANNM', 'MODA', 'MODAFM']:
        model = MODANet().to(device)
    else:
        raise ValueError('Unknown model {}'.format(cfg['model']))

    if cfg['model'] != 'DANNS':
        model.load_state_dict(torch.load(cfg['input']))
        accuracies, losses = test_routine(model, test_loaders, cfg)
        print('target pub: acc = {:.3f},'.format(accuracies['target pub']), 'loss = {:.3f}'.format(losses['target pub']))
        print('target priv: acc = {:.3f},'.format(accuracies['target priv']), 'loss = {:.3f}'.format(losses['target priv']))

    else:  # for DANNS, report results for the best source domain
        src_domains = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits']
        src_domains.remove(cfg['target'])
        for i, src in enumerate(src_domains):
            model.load_state_dict(torch.load(cfg['input']+'_'+src))
            acc, loss = test_routine(model, test_loaders, cfg)
            if i == 0:
                accuracies = acc
                losses = loss
            else:
                for key in accuracies.keys():
                    accuracies[key] = acc[key] if (acc[key] > accuracies[key]) else accuracies[key]
                    losses[key] = loss[key] if (acc[key] > accuracies[key]) else losses[key]
        log.print('target pub: acc = {:.3f},'.format(accuracies['target pub']), 'loss = {:.3f}'.format(losses['target pub']), level=1)
        log.print('target priv: acc = {:.3f},'.format(accuracies['target priv']), 'loss = {:.3f}'.format(losses['target priv']), level=1)


if __name__ == '__main__':
    main()
