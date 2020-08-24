import sys
sys.path.append('..')

import argparse
from configparser import ConfigParser
import random
import ast

import numpy as np
import torch
from torch.utils.data import Subset

from dataset import Amazon
from models import MDANet, MODANet
from routines import (cross_validation, mdan_train_routine,
                      moda_train_routine, moda_mlp_fm_train_routine)
from utils import Logger


def main():
    # N.B.: parameters defined in cv_cfg.ini override args!
    parser = argparse.ArgumentParser(description='Cross-validation over source domains for the Amazon dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MODAFM', type=str, metavar='', help='model type (\'MDAN\' / \'MODA\' / \'MODAFM\'')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/Amazon', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='books', type=str, metavar='', help='target domain (\'books\' / \'dvd\' / \'electronics\' / \'kitchen\')')
    parser.add_argument('-o', '--output', default='msda_hyperparams.ini', type=str, metavar='', help='output file')
    parser.add_argument('-n', '--n_iter', default=20, type=int, metavar='', help='number of CV iterations')
    parser.add_argument('--n_samples', default=2000, type=int, metavar='', help='number of samples from each domain')
    parser.add_argument('--n_features', default=5000, type=int, metavar='', help='number of features to use')
    parser.add_argument('--mu', type=float, default=1e-2, help="hyperparameter of the coefficient for the domain adversarial loss")
    parser.add_argument('--beta', type=float, default=2e-1, help="hyperparameter of the non-sparsity regularization")
    parser.add_argument('--lambda', type=float, default=1e-1, help="hyperparameter of the FixMatch loss")
    parser.add_argument('--min_dropout', type=int, default=2e-1, help="minimum dropout rate")
    parser.add_argument('--max_dropout', type=int, default=8e-1, help="maximum dropout rate")
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='hyperparameter of weight decay regularization')
    parser.add_argument('--lr', default=1e0, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=15, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=20, type=int, metavar='', help='batch size (per domain)')
    parser.add_argument('--checkpoint', default=0, type=int, metavar='', help='number of epochs between saving checkpoints (0 disables checkpoints)')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='amazon_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--verbosity', default=2, type=int, metavar='', help='log verbosity level')
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

    domains = ['books', 'dvd', 'electronics', 'kitchen']
    datasets = {}
    for domain in domains:
        if domain == cfg['target']:
            continue
        datasets[domain] = Amazon('./amazon.npz', domain, dimension=cfg['n_features'], transform=torch.from_numpy)
        indices = random.sample(range(len(datasets[domain])), cfg['n_samples'])
        datasets[domain] = Subset(datasets[domain], indices)
    cfg['test_transform'] = torch.from_numpy

    if cfg['model'] == 'MDAN':
        model = MDANet(input_dim=cfg['n_features'], n_classes=2, n_domains=len(domains)-2).to(device)
        cfg['model'] = model
        cfg['train_routine'] = lambda model, optimizer, train_loader, cfg: mdan_train_routine(model, optimizer, train_loader, dict(), cfg)
    elif cfg['model'] == 'MODA':
        model = MODANet(input_dim=cfg['n_features'], n_classes=2).to(device)
        cfg['model'] = model
        cfg['train_routine'] = lambda model, optimizer, train_loader, cfg: moda_train_routine(model, optimizer, train_loader, dict(), cfg)
    elif cfg['model'] == 'MODAFM':
        model = MODANet(input_dim=cfg['n_features'], n_classes=2).to(device)
        cfg['model'] = model
        cfg['train_routine'] = lambda model, optimizer, train_loader, cfg: moda_mlp_fm_train_routine(model, optimizer, train_loader, dict(), cfg)

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
