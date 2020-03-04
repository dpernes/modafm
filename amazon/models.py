import sys
sys.path.append('..')

from collections import OrderedDict
import torch.nn as nn

from gradient_reversal import GradientReversalLayer

class MDANet(nn.Module):
    def __init__(self, input_dim, n_classes, n_domains):
        super(MDANet, self).__init__()
        self.feat_ext = nn.Sequential(OrderedDict([
            ('Linear1', nn.Linear(input_dim, 1000)),
            ('ReLU1', nn.ReLU()),
            ('Linear2', nn.Linear(1000, 500)),
            ('ReLU2', nn.ReLU()),
            ('Linear3', nn.Linear(500, 100)),
            ('ReLU3', nn.ReLU())
        ]))

        self.task_class = nn.Linear(100, n_classes)
        self.grad_reverse = nn.ModuleList([GradientReversalLayer() for _ in range(n_domains)])
        self.domain_class = nn.ModuleList([nn.Linear(100, 2) for _ in range(n_domains)])

    def forward(self, Xs, Xt):
        ys, ds, dt = [], [], []
        for i in range(len(self.domain_class)):
            # process source data
            Z = self.feat_ext(Xs[i])
            ys.append(self.task_class(Z))
            ds.append(self.domain_class[i](self.grad_reverse[i](Z)))

            # process target data
            Z = self.feat_ext(Xt)
            dt.append(self.domain_class[i](self.grad_reverse[i](Z)))

        return ys, ds, dt

    def inference(self, X):
        Z = self.feat_ext(X)
        y = self.task_class(Z)

        return y


class MixMDANet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(MixMDANet, self).__init__()
        self.feat_ext = nn.Sequential(OrderedDict([
            ('Linear1', nn.Linear(input_dim, 1000)),
            ('ReLU1', nn.ReLU()),
            ('Linear2', nn.Linear(1000, 500)),
            ('ReLU2', nn.ReLU()),
            ('Linear3', nn.Linear(500, 100)),
            ('ReLU3', nn.ReLU())
        ]))

        self.task_class = nn.Linear(100, n_classes)
        self.grad_reverse = GradientReversalLayer()
        self.domain_class = nn.Linear(100, 2)

        self.feat_ext = nn.Sequential(OrderedDict([
            ('Linear1', nn.Linear(input_dim, 500)),
            ('ReLU1', nn.ReLU()),
        ]))

        self.task_class = nn.Linear(500, n_classes)
        self.grad_reverse = GradientReversalLayer()
        self.domain_class = nn.Linear(500, 2)

    def forward(self, Xs, Xt):
        ys, ds = [], []
        for Xsi in Xs:
            # process source data
            Z = self.feat_ext(Xsi)
            ys.append(self.task_class(Z))
            ds.append(self.domain_class(self.grad_reverse(Z)))

        # process target data
        Z = self.feat_ext(Xt)
        dt = self.domain_class(self.grad_reverse(Z))

        return ys, ds, dt

    def inference(self, X):
        Z = self.feat_ext(X)
        y = self.task_class(Z)

        return y