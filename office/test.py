import sys
sys.path.append('..')

import argparse
from configparser import ConfigParser
import ast
import random
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dataset import Office
from models import SimpleCNN, MDANet, MODANet
from routines import test_routine
from utils import MSDA_Loader, Logger


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with Office dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MODAFM', type=str, metavar='', help='model type (\'FS\' / \'DANNS\' / \'DANNM\' / \'MDAN\' / \'MODA\' / \'FM\' / \'MODAFM\'')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/OfficeRsz', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='amazon', type=str, metavar='', help='target domain (\'amazon\' / \'dslr\' / \'webcam\')')
    parser.add_argument('-i', '--input', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--arch', default='resnet50', type=str, metavar='', help='network architecture (\'resnet50\' / \'alexnet\'')
    parser.add_argument('--batch_size', default=20, type=int, metavar='', help='batch size (per domain)')
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

    # normalization transformation (required for pretrained networks)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if 'FM' in cfg['model']:
        transform = T.ToTensor()
    else:
        transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])

    domains = ['amazon', 'dslr', 'webcam']
    datasets = {domain: Office(cfg['data_path'], domain=domain, transform=transform) for domain in domains}
    n_classes = len(datasets[cfg['target']].class_names)

    if 'FM' in cfg['model']:
        test_set = deepcopy(datasets[cfg['target']])
        test_set.transform = T.ToTensor()  # no data augmentation in test
    else:
        test_set = datasets[cfg['target']]

    if cfg['model'] != 'FS':
        test_loader = {'target pub': DataLoader(test_set, batch_size=3*cfg['batch_size'])}
    else:
        train_indices = random.sample(range(len(datasets[cfg['target']])), int(0.8*len(datasets[cfg['target']])))
        test_indices = list(set(range(len(datasets[cfg['target']]))) - set(train_indices))
        test_loader = {'target pub': DataLoader(
            datasets[cfg['target']],
            batch_size=cfg['batch_size'],
            sampler=SubsetRandomSampler(test_indices))}
    log.print('target domain:', cfg['target'], level=1)

    if cfg['model'] in ['FS', 'FM']:
        model = SimpleCNN(n_classes=n_classes, arch=cfg['arch']).to(device)
    elif args['model'] == 'MDAN':
        model = MDANet(n_classes=n_classes, n_domains=len(domains)-1, arch=cfg['arch']).to(device)
    elif cfg['model'] in ['DANNS', 'DANNM', 'MODA', 'MODAFM']:
        model = MODANet(n_classes=n_classes, arch=cfg['arch']).to(device)
    else:
        raise ValueError('Unknown model {}'.format(cfg['model']))

    if cfg['model'] != 'DANNS':
        model.load_state_dict(torch.load(cfg['input']))
        accuracies, losses = test_routine(model, test_loader, cfg)
        print('target pub: acc = {:.3f},'.format(accuracies['target pub']), 'loss = {:.3f}'.format(losses['target pub']))

    else:  # for DANNS, report results for the best source domain
        src_domains = ['amazon', 'dslr', 'webcam']
        src_domains.remove(cfg['target'])
        for i, src in enumerate(src_domains):
            model.load_state_dict(torch.load(cfg['input']+'_'+src))
            acc, loss = test_routine(model, test_loader, cfg)
            if i == 0:
                accuracies = acc
                losses = loss
            else:
                for key in accuracies.keys():
                    accuracies[key] = acc[key] if (acc[key] > accuracies[key]) else accuracies[key]
                    losses[key] = loss[key] if (acc[key] > accuracies[key]) else losses[key]
        log.print('target pub: acc = {:.3f},'.format(accuracies['target pub']), 'loss = {:.3f}'.format(losses['target pub']), level=1)

if __name__ == '__main__':
    main()
