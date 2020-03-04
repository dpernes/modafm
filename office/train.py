import sys
sys.path.append('..')

import argparse
import random
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import Office
from models import MDANet, MixMDANet
from routines import mdan_train_routine, mixmdan_train_routine, mixmdan_fm_train_routine
from utils import MSDA_Loader


def main():
    parser = argparse.ArgumentParser(description='Domain adaptation experiments with Office dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='MDAN', type=str, metavar='', help='model type (\'MDAN\' / \'MDANU\' / \'MDANFM\' / \'MDANUFM\' / \'MixMDAN\' / \'MixMDANFM\')')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/OfficeRsz', type=str, metavar='', help='data directory path')
    parser.add_argument('-t', '--target', default='amazon', type=str, metavar='', help='target domain (\'amazon\' / \'dslr\' / \'webcam\')')
    parser.add_argument('-o', '--output', default='msda.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('--mode', default='dynamic', type=str, metavar='', help='mode of combination rule (\'dynamic\' / \'minmax\')')
    parser.add_argument('--mu', type=float, default=1e-2, help="hyperparameter of the coefficient for the domain adversarial loss")
    parser.add_argument('--gamma', type=float, default=10., help="hyperparameter of the dynamic loss")
    parser.add_argument('--beta', type=float, default=0.2, help="hyperparameter of the non-sparsity regularization")
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

    # normalization transformation (required for pretrained networks)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if 'FM' in args['model']:
        # weak data augmentation (small rotation + small translation)
        data_aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(5, translate=(0.125, 0.125)),
            T.ToTensor(),
            # normalize,
        ])
    else:
        data_aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

    domains = ['amazon', 'dslr', 'webcam']
    datasets = {domain: Office(args['data_path'], domain=domain, transform=data_aug) for domain in domains}
    n_classes = len(datasets[args['target']].class_names)

    if 'FM' in args['model']:
        test_set = deepcopy(datasets[args['target']])
        test_set.transform = T.ToTensor()  # no data augmentation in test
    else:
        test_set = datasets[args['target']]

    train_loader = MSDA_Loader(datasets, args['target'], batch_size=args['batch_size'], shuffle=True, num_workers=0, device=device)
    test_loader = DataLoader(test_set, batch_size=3*args['batch_size'])
    valid_loaders = {'pub target': test_loader}
    print('source domains:', train_loader.sources)

    args['excl_rand_aug'] = None
    if args['model'] == 'MDAN':
        model = MDANet(n_classes=n_classes, n_domains=len(train_loader.sources)).to(device)

        conv_params, fc_params = [], []
        for name, param in model.named_parameters():
            if 'FC' in name.upper():
                fc_params.append(param)
            else:
                conv_params.append(param)
        optimizer = optim.Adadelta([
            {'params':conv_params, 'lr':0.1*args['lr'], 'weight_decay':args['weight_decay']},
            {'params':fc_params, 'lr':args['lr'], 'weight_decay':args['weight_decay']}
        ])

        mdan_train_routine(model, optimizer, train_loader, valid_loaders, args)
    elif args['model'] == 'MixMDAN':
        model = MixMDANet(n_classes=n_classes).to(device)

        conv_params, fc_params = [], []
        for name, param in model.named_parameters():
            if 'FC' in name.upper():
                fc_params.append(param)
            else:
                conv_params.append(param)
        optimizer = optim.Adadelta([
            {'params':conv_params, 'lr':0.1*args['lr'], 'weight_decay':args['weight_decay']},
            {'params':fc_params, 'lr':args['lr'], 'weight_decay':args['weight_decay']}
        ])

        mixmdan_train_routine(model, optimizer, train_loader, valid_loaders, args)
    elif args['model'] == 'MixMDANFM':
        model = MixMDANet(n_classes=n_classes).to(device)

        conv_params, fc_params = [], []
        for name, param in model.named_parameters():
            if 'FC' in name.upper():
                fc_params.append(param)
            else:
                conv_params.append(param)
        optimizer = optim.Adadelta([
            {'params':conv_params, 'lr':0.1*args['lr'], 'weight_decay':args['weight_decay']},
            {'params':fc_params, 'lr':args['lr'], 'weight_decay':args['weight_decay']}
        ])

        mixmdan_fm_train_routine(model, optimizer, train_loader, valid_loaders, args)
    else:
        raise ValueError('Unknown model {}'.format(args['model']))

    torch.save(model.state_dict(), args['output'])

if __name__ == '__main__':
    main()
