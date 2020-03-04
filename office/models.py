import sys
sys.path.append('..')

from collections import OrderedDict
import torch
from torch import nn
import torchvision.models as models

from gradient_reversal import GradientReversalLayer

class Flatten(nn.Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)

class MDANet(nn.Module):
    def __init__(self, n_domains, n_classes):
        super(MDANet, self).__init__()

        resnet_layers = dict(models.resnet50(pretrained=True).named_children())
        self.feat_ext = nn.Sequential(OrderedDict(
            [(key, resnet_layers[key])
             for key in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']]))

        self.task_class = nn.Sequential(
            OrderedDict(
                [(key, resnet_layers[key])
                 for key in ['layer4', 'avgpool']]
                +[('flatten', Flatten()), ('fc', nn.Linear(2048, n_classes))]))

        self.grad_reverse = nn.ModuleList([GradientReversalLayer() for _ in range(n_domains)])

        self.domain_class = nn.ModuleList([])
        for _ in range(n_domains):
            resnet_layers = dict(models.resnet50(pretrained=True).named_children())
            self.domain_class.append(nn.Sequential(
                OrderedDict(
                    [(key, resnet_layers[key])
                     for key in ['layer4', 'avgpool']]
                    +[('flatten', Flatten()), ('fc', nn.Linear(2048, 2))])))


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
    def __init__(self, n_classes):
        super(MixMDANet, self).__init__()

        resnet_layers = dict(models.resnet50(pretrained=True).named_children())
        self.feat_ext = nn.Sequential(OrderedDict(
            [(key, resnet_layers[key])
             for key in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']]))

        self.task_class = nn.Sequential(
            OrderedDict(
                [(key, resnet_layers[key])
                 for key in ['layer4', 'avgpool']]
                +[('flatten', Flatten()), ('fc', nn.Linear(2048, n_classes))]))

        self.grad_reverse = GradientReversalLayer()

        resnet_layers = dict(models.resnet50(pretrained=True).named_children())
        self.domain_class = nn.Sequential(
            OrderedDict(
                [(key, resnet_layers[key])
                 for key in ['layer4', 'avgpool']]
                +[('flatten', Flatten()), ('fc', nn.Linear(2048, 2))]))

    def forward(self, Xs, Xt):
        ys, ds = [], []
        for i in range(len(Xs)):
            # process source data
            Z = self.feat_ext(Xs[i])
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


if __name__ == '__main__':
    N = 32
    model = MDANet(n_classes=31, n_domains=3)
    print(model)
    Xs = [torch.zeros((N, 3, 256, 256)) for _ in range(3)]
    Xt = torch.zeros((N, 3, 256, 256))
    yt = model.inference(Xt)
    ys, ds, dt = model(Xs, Xt)
    print('ys', ys[0].shape, 'ds', ds[0].shape, 'yt', yt.shape, 'dt', dt[0].shape)
