import sys
sys.path.append('..')

import argparse
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import Subset, DataLoader

from dataset import Amazon
from models import MDANet, MixMDANet
from routines import mdan_train_routine, mixmdan_train_routine
from utils import MSDA_Loader


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with Amazon dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MDAN', type=str, metavar='', help='model type (\'MDAN\' / \'MDANU\' / \'MDANFM\' / \'MDANUFM\' / \'MixMDAN\' / \'MixMDANFM\')')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/Amazon', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='books', type=str, metavar='', help='target domain (\'books\' / \'dvd\' / \'electronics\' / \'kitchen\')')
    parser.add_argument('-o', '--output', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--mode', default='dynamic', type=str, metavar='', help='mode of combination rule (\'dynamic\' / \'minmax\')')
    parser.add_argument('--n_samples', default=2000, type=int, metavar='', help='number of samples from each domain')
    parser.add_argument('--n_features', default=5000, type=int, metavar='', help='number of features to use')
    parser.add_argument('--mu', type=float, default=1e-2, help="hyperparameter of the coefficient for the domain adversarial loss")
    parser.add_argument('--gamma', type=float, default=10., help="hyperparameter of the dynamic loss")
    parser.add_argument('--beta', type=float, default=0.2, help="hyperparameter of the non-sparsity regularization")
    parser.add_argument('--lambda', type=float, default=1e-1, help="hyperparameter of the FixMatch loss")
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='hyperparameter of weight decay regularization')
    parser.add_argument('--lr', default=1e0, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=15, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=20, type=int, metavar='', help='batch size (per domain)')
    parser.add_argument('--checkpoint', default=0, type=int, metavar='', help='number of epochs between saving checkpoints (0 disables checkpoints)')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='amazon_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())

    device = 'cuda' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    print('device:', device)

    # dump args to a txt file for your records
    with open(args['output'] + '.txt', 'w') as f:
        f.write(str(args)+'\n')

    # use a fixed random seed for reproducibility purposes
    if args['seed'] > 0:
        random.seed(args['seed'])
        np.random.seed(seed=args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])

    products = ['books', 'dvd', 'electronics', 'kitchen']
    datasets = {}
    for product in products:
        datasets[product] = Amazon('./amazon.npz', product, dimension=args['n_features'], transform=torch.from_numpy)
        indices = random.sample(range(len(datasets[product])), args['n_samples'])
        if product == args['target']:
            priv_indices = list(set(range(len(datasets[args['target']]))) - set(indices))
            test_priv_set = Subset(datasets[args['target']], priv_indices)
        datasets[product] = Subset(datasets[product], indices)
    test_pub_set = datasets[args['target']]

    train_loader = MSDA_Loader(datasets, args['target'], batch_size=args['batch_size'], shuffle=True, device=device)
    test_pub_loader = DataLoader(test_pub_set, batch_size=4*args['batch_size'])
    test_priv_loader = DataLoader(test_priv_set, batch_size=4*args['batch_size'])
    valid_loaders = {'pub target': test_pub_loader, 'priv target': test_priv_loader}
    print('source domains:', train_loader.sources)

    if args['model'] == 'MDAN':
        model = MDANet(input_dim=args['n_features'], n_classes=2, n_domains=len(train_loader.sources)).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        mdan_train_routine(model, optimizer, train_loader, valid_loaders, args)
    elif args['model'] == 'MixMDAN':
        model = MixMDANet(input_dim=args['n_features'], n_classes=2).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        mixmdan_train_routine(model, optimizer, train_loader, valid_loaders, args)

    torch.save(model.state_dict(), args['output'])

if __name__ == '__main__':
    main()
