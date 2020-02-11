from collections import OrderedDict

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lbd):
        ctx.save_for_backward(lbd)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        lbd, = ctx.saved_tensors
        return -lbd*grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lbd=1.):
        super(GradientReversalLayer, self).__init__()
        self.lbd = nn.Parameter(torch.FloatTensor([lbd]), requires_grad=False)
        self.grad_reverse = GradientReversal()

    def forward(self, x):
        return self.grad_reverse.apply(x, self.lbd)

class Flatten(nn.Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)

class MDANet(nn.Module):
    def __init__(self, n_domains=3):
        super(MDANet, self).__init__()

        self.feat_ext = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('ReLU1', nn.ReLU()),
            ('MaxPool1', nn.MaxPool2d(2)),
            ('Conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('ReLU2', nn.ReLU()),
            ('MaxPool2', nn.MaxPool2d(2)),
            ('Conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ('ReLU3', nn.ReLU()),
        ]))

        self.task_class = nn.Sequential(OrderedDict([
            ('MaxPool3', nn.MaxPool2d(2)),
            ('Conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('ReLU4', nn.ReLU()),
            ('Flatten', Flatten()),
            ('FC1', nn.Linear(4096, 2048)),  # 4096 = (256*32*32)/(2^3*2^3)
            ('ReLU5', nn.ReLU()),
            ('FC2', nn.Linear(2048, 1024)),
            ('ReLU6', nn.ReLU()),
            ('FC3', nn.Linear(1024, 10)),
        ]))

        self.grad_reverse = nn.ModuleList([GradientReversalLayer() for _ in range(n_domains)])
        self.domain_class = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(2),
                Flatten(),
                nn.Linear(4096, 2048),  # 4096 = (256*32*32)/(2^3*2^3)
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2),
            ) for _ in range(n_domains)])

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


if __name__ == '__main__':
    N = 32
    model = MDANet()
    Xs = [torch.zeros((N, 3, 32, 32)) for _ in range(3)]
    Xt = torch.zeros((N, 3, 32, 32))
    yt = model.inference(Xt)
    ys, ds, dt = model(Xs, Xt)
    print('ys', ys[0].shape, 'ds', ds[0].shape, 'yt', yt.shape, 'dt', dt[0].shape)
