import sys
sys.path.append('..')

import argparse
from configparser import ConfigParser
import ast
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import Subset, DataLoader

from dataset import Amazon
from models import SimpleMLP, MDANet, MODANet
from routines import test_routine
from utils import MSDA_Loader, Logger


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with Amazon dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MODAFM', type=str, metavar='', help='model type (\'FS\' / \'DANNS\' / \'DANNM\' / \'MDAN\' / \'MODA\' / \'FM\' / \'MODAFM\'')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/Amazon', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='books', type=str, metavar='', help='target domain (\'books\' / \'dvd\' / \'electronics\' / \'kitchen\')')
    parser.add_argument('-i', '--input', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--n_samples', default=2000, type=int, metavar='', help='number of samples from each domain')
    parser.add_argument('--n_features', default=5000, type=int, metavar='', help='number of features to use')
    parser.add_argument('--batch_size', default=20, type=int, metavar='', help='batch size (per domain)')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--verbosity', default=2, type=int, metavar='', help='log verbosity level (0, 1, 2)')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())
    cfg = args.copy()


    device = 'cuda' if (cfg['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    log = Logger(cfg['verbosity'])
    log.print('device:', device, level=0)

    # use a fixed random seed for reproducibility purposes
    if cfg['seed'] > 0:
        random.seed(args['seed'])
        np.random.seed(seed=args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])

    domains = ['books', 'dvd', 'electronics', 'kitchen']
    datasets = {}
    for domain in domains:
        datasets[domain] = Amazon('./amazon.npz', domain, dimension=cfg['n_features'], transform=torch.from_numpy)
        indices = random.sample(range(len(datasets[domain])), cfg['n_samples'])
        if domain == cfg['target']:
            priv_indices = list(set(range(len(datasets[cfg['target']]))) - set(indices))
            test_priv_set = Subset(datasets[cfg['target']], priv_indices)
        datasets[domain] = Subset(datasets[domain], indices)
    test_pub_set = datasets[cfg['target']]

    test_pub_loader = DataLoader(test_pub_set, batch_size=4*cfg['batch_size'])
    test_priv_loader = DataLoader(test_priv_set, batch_size=4*cfg['batch_size'])
    test_loaders = {'target pub': test_pub_loader, 'target priv': test_priv_loader}
    log.print('target domain:', cfg['target'], level=1)

    if cfg['model'] in ['FS', 'FM']:
        model = SimpleMLP(input_dim=cfg['n_features'], n_classes=2).to(device)
    elif cfg['model'] in ['MDAN', 'MDANFM']:
        model = MDANet(input_dim=cfg['n_features'], n_classes=2, n_domains=len(datasets)-1).to(device)
    elif cfg['model'] in ['DANNS', 'DANNM', 'MODA', 'MODAFM']:
        model = MODANet(input_dim=cfg['n_features'], n_classes=2).to(device)
    else:
        raise ValueError('Unknown model {}'.format(cfg['model']))

    if cfg['model'] != 'DANNS':
        model.load_state_dict(torch.load(cfg['input']))
        accuracies, losses = test_routine(model, test_loaders, cfg)
        print('target pub: acc = {:.3f},'.format(accuracies['target pub']), 'loss = {:.3f}'.format(losses['target pub']))
        print('target priv: acc = {:.3f},'.format(accuracies['target priv']), 'loss = {:.3f}'.format(losses['target priv']))

    else:  # for DANNS, report results for the best source domain
        domains.remove(cfg['target'])
        for i, src in enumerate(domains):
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
