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
