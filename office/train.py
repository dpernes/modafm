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
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dataset import Office
from models import MDANet, MixMDANet
from routines import mdan_train_routine, mixmdan_train_routine, mixmdan_fm_train_routine
from utils import MSDA_Loader, Logger


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with Office dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MDAN', type=str, metavar='', help='model type (\'MDAN\' / \'MDANU\' / \'MDANFM\' / \'MDANUFM\' / \'MixMDAN\' / \'MixMDANFM\')')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/OfficeRsz', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='amazon', type=str, metavar='', help='target domain (\'amazon\' / \'dslr\' / \'webcam\')')
    parser.add_argument('-o', '--output', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--icfg', default=None, type=str, metavar='', help='config file (overrides args)')
    parser.add_argument('--mode', default='dynamic', type=str, metavar='', help='mode of combination rule (\'dynamic\' / \'minmax\')')
    parser.add_argument('--mu', type=float, default=1e-2, help="hyperparameter of the coefficient for the domain adversarial loss")
    parser.add_argument('--beta', type=float, default=0., help="hyperparameter of the non-sparsity regularization")
    parser.add_argument('--lambda', type=float, default=1e-1, help="hyperparameter of the FixMatch loss")
    parser.add_argument('--n_rand_aug', type=int, default=2, help="N parameter of RandAugment")
    parser.add_argument('--m_min_rand_aug', type=int, default=3, help="minimum M parameter of RandAugment")
    parser.add_argument('--m_max_rand_aug', type=int, default=10, help="maximum M parameter of RandAugment")
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='hyperparameter of weight decay regularization')
    parser.add_argument('--lr', default=1e-1, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=15, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=20, type=int, metavar='', help='batch size (per domain)')
    parser.add_argument('--checkpoint', default=0, type=int, metavar='', help='number of epochs between saving checkpoints (0 disables checkpoints)')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='office_train', type=str, metavar='', help='Visdom environment name')
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

    # dump args to a txt file for your records
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

    # normalization transformation (required for pretrained networks)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if 'FM' in cfg['model']:
        # weak data augmentation (small rotation + small translation)
        data_aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(5, translate=(0.125, 0.125)),
            T.ToTensor(),
            # normalize,  # normalization disrupts FixMatch
        ])
    else:
        data_aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

    domains = ['amazon', 'dslr', 'webcam']
    datasets = {domain: Office(cfg['data_path'], domain=domain, transform=data_aug) for domain in domains}
    n_classes = len(datasets[cfg['target']].class_names)

    if 'FM' in cfg['model']:
        test_set = deepcopy(datasets[cfg['target']])
        test_set.transform = T.ToTensor()  # no data augmentation in test
    else:
        test_set = datasets[cfg['target']]

    train_loader = MSDA_Loader(datasets, cfg['target'], batch_size=cfg['batch_size'], shuffle=True, num_workers=0, device=device)
    test_loader = DataLoader(test_set, batch_size=3*cfg['batch_size'])
    valid_loaders = {'pub target': test_loader}
    log.print('target domain:', cfg['target'], 'source domains:', train_loader.sources, level=1)

    if args['model'] == 'MDAN':
        model = MDANet(n_classes=n_classes, n_domains=len(train_loader.sources)).to(device)

        conv_params, fc_params = [], []
        for name, param in model.named_parameters():
            if 'FC' in name.upper():
                fc_params.append(param)
            else:
                conv_params.append(param)
        optimizer = optim.Adadelta([
            {'params':conv_params, 'lr':0.1*cfg['lr'], 'weight_decay':cfg['weight_decay']},
            {'params':fc_params, 'lr':cfg['lr'], 'weight_decay':cfg['weight_decay']}
        ])

        mdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MixMDAN':
        model = MixMDANet(n_classes=n_classes).to(device)

        conv_params, fc_params = [], []
        for name, param in model.named_parameters():
            if 'FC' in name.upper():
                fc_params.append(param)
            else:
                conv_params.append(param)
        optimizer = optim.Adadelta([
            {'params':conv_params, 'lr':0.1*cfg['lr'], 'weight_decay':cfg['weight_decay']},
            {'params':fc_params, 'lr':cfg['lr'], 'weight_decay':cfg['weight_decay']}
        ])

        mixmdan_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    elif cfg['model'] == 'MixMDANFM':
        model = MixMDANet(n_classes=n_classes).to(device)

        conv_params, fc_params = [], []
        for name, param in model.named_parameters():
            if 'FC' in name.upper():
                fc_params.append(param)
            else:
                conv_params.append(param)
        optimizer = optim.Adadelta([
            {'params':conv_params, 'lr':0.1*cfg['lr'], 'weight_decay':cfg['weight_decay']},
            {'params':fc_params, 'lr':cfg['lr'], 'weight_decay':cfg['weight_decay']}
        ])

        cfg['excl_transf'] = None
        mixmdan_fm_train_routine(model, optimizer, train_loader, valid_loaders, cfg)
    else:
        raise ValueError('Unknown model {}'.format(cfg['model']))

    torch.save(model.state_dict(), cfg['output'])

if __name__ == '__main__':
    main()
