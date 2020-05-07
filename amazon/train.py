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
from models import SimpleMLP, MDANet, MixMDANet
from routines import (fs_train_routine, mlp_fm_train_routine, dann_train_routine, mdan_train_routine,
                      mixmdan_train_routine, mixmdan_mlp_fm_train_routine)
from utils import MSDA_Loader, Logger


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with Amazon dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MDAN', type=str, metavar='', help='model type (\'MDAN\' / \'MDANU\' / \'MDANFM\' / \'MDANUFM\' / \'MixMDAN\' / \'MixMDANFM\')')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/Amazon', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='books', type=str, metavar='', help='target domain (\'books\' / \'dvd\' / \'electronics\' / \'kitchen\')')
    parser.add_argument('-o', '--output', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--icfg', default=None, type=str, metavar='', help='config file (overrides args)')
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
    parser.add_argument('--eval_target', default=False, type=int, metavar='', help='evaluate target during training')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='amazon_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--verbosity', default=2, type=int, metavar='', help='log verbosity level')
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

    train_loader = MSDA_Loader(datasets, cfg['target'], batch_size=cfg['batch_size'], shuffle=True, device=device)
    test_pub_loader = DataLoader(test_pub_set, batch_size=4*cfg['batch_size'])
    test_priv_loader = DataLoader(test_priv_set, batch_size=4*cfg['batch_size'])
    valid_loaders = {'target pub': test_pub_loader, 'target priv': test_priv_loader} if cfg['eval_target'] else None
    log.print('target domain:', cfg['target'], 'source domains:', train_loader.sources, level=1)

    if cfg['model'] == 'FS':
        model = SimpleMLP(input_dim=cfg['n_features'], n_classes=2).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        if cfg['eval_target']:
            del valid_loaders['target pub']
        fs_train_routine(model, optimizer, test_pub_loader, valid_loaders, cfg)
    elif cfg['model'] == 'FM':
        model = SimpleMLP(input_dim=cfg['n_features'], n_classes=2).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        mlp_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'DANNS':
        for src in train_loader.sources:
            model = MixMDANet(input_dim=cfg['n_features'], n_classes=2).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            dataset_ss = {src: datasets[src], cfg['target']: datasets[cfg['target']]}
            train_loader = MSDA_Loader(dataset_ss, cfg['target'], batch_size=cfg['batch_size'], shuffle=True, device=device)
            dann_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
            torch.save(model.state_dict(), cfg['output']+'_'+src)
    elif cfg['model'] == 'DANNM':
        model = MixMDANet(input_dim=cfg['n_features'], n_classes=2).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        dann_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MDAN':
        model = MDANet(input_dim=cfg['n_features'], n_classes=2, n_domains=len(train_loader.sources)).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        mdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MixMDAN':
        model = MixMDANet(input_dim=cfg['n_features'], n_classes=2).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args['lr'], weight_decay=cfg['weight_decay'])
        mixmdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MixMDANFM':
        model = MixMDANet(input_dim=cfg['n_features'], n_classes=2).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        mixmdan_mlp_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg)

    torch.save(model.state_dict(), cfg['output'])

if __name__ == '__main__':
    main()
