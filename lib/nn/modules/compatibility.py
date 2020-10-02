import torch
from torch import nn
from .. import functional as LF

def _get_compatibility_module(compatibility, *args, **kwargs):
    if compatibility == 'multiplicative':
        return MulComp(*args, **kwargs)
    elif compatibility == 'additive':
        return AddComp(*args, **kwargs)
    raise ValueError("{} is not a valid compatibility mode".format(compatibility))

def _get_compatibility_activation(activation, *args, **kwargs):
    if activation == 'tanh':
        return nn.Tanh(*args, **kwargs)
    elif activation == 'sigmoid':
        return nn.Sigmoid(*args, **kwargs)
    raise ValueError("{} is not a valid activation for compatibility mode".format(activation))

# un-parameterized
class MulComp(nn.Module):
    """
    Computes: f(q, k) = q * k.T
    """
    def __init__(self, input_size=None):
        super(MulComp, self).__init__()
        # input_size ignored; only present to keep a standard interface
        
    def forward(self, queries, keys):  
        return LF.mulcomp(queries, keys)

# parameterized
class AddComp(nn.Module):
    """
    Computes: f(q, k) = v.T * act(q + k)
    """
    def __init__(self, input_size, activation='tanh'):
        super(AddComp, self).__init__()
        self.act = _get_compatibility_activation(activation)
        self.v = nn.Parameter(torch.Tensor(input_size)) # importance vector
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 1 / self.v.shape[0]**.5
        self.v.data.uniform_(-bound, bound)

    def forward(self, queries, keys):
        """
        queries : (*, m, H)
        keys : (*, n, H)
        out : (*, m, n) subject to broadcasting
        """
        Q  = queries.unsqueeze(-2)        # (*, m, 1, H)
        K  = keys.unsqueeze(-3)           # (*, 1, n, H)
        QK = self.act(Q + K)              # (*, m, n, H)
        logits = torch.matmul(QK, self.v) # (*, m, n)
        return logits