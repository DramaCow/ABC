import torch
from torch import nn
import torch.nn.functional as F

class ScaledSoftmax(nn.Module):
    def __init__(self, scalar, dim=None):
        super(ScaledSoftmax, self).__init__()
        self.scalar = scalar
        self.dim   = dim

    def forward(self, input):
        return F.softmax(self.scalar * input, self.dim, _stacklevel=5)