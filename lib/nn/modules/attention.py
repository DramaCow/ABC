import torch
from torch import nn
from .. import functional as LF
import lib.nn as lnn
from .compatibility import _get_compatibility_module

class Attention(nn.Module):
    r"""
    Args:
        query_size: size of each query sample
        key_size: size of each key sample
        value_size: size of each value sample
        num_heads: number of attention heads. Default: 1
        head_size: size of each individual attention head. Default: value_size / num_heads
        embed_values: if enabled, values are passed through a linear layer before attention
            is applied
        reduce: if enabled, the outputs of the attention heads are transformed to
            a value_size vector
    """
    def __init__(self, 
                 query_size,
                 key_size,
                 value_size,
                 num_heads=1,
                 head_size=None,
                 symmetric=False,
                 embed_values=True,
                 compatibility='multiplicative',
                 compatibility_config={},
                 activation='softmax',
                 reduce=True):
        super(Attention, self).__init__()

        self.num_heads          = num_heads
        self.embed_values       = embed_values
        self.reduce             = reduce

        if head_size is None:
            head_size = value_size // num_heads

        if symmetric:
            fc = nn.Linear(query_size, num_heads * head_size)
            self.fc_q = self.fc_k = fc
        else:
            self.fc_q = nn.Linear(query_size, num_heads * head_size)
            self.fc_k = nn.Linear(key_size, num_heads * head_size)

        self.compat = _get_compatibility_module(compatibility, head_size, **compatibility_config)
        self.act    = self._get_attention_activation(activation, query_size**-.5, dim=-1)

        if embed_values:
            self.fc_v = nn.Linear(value_size, num_heads * head_size)
            self.fc_o = nn.Linear(num_heads * head_size, value_size) if reduce else None
        else:
            self.fc_o = nn.Linear(num_heads * value_size, value_size) if reduce else None
        
    def forward(self, queries, keys, values):
        """
        Q : (*, m, q)
        K : (*, n, k)
        V : (*, n, v)
        
        let h = num_heads
            d = head_size

        High-dimensions (*) subject to broadcasting.
        """
        
        # project then reshape queries and keys (such tha num_heads dimension comes first)
        Q = self.fc_q(queries)                                             # (*, m, h*d)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, -1).transpose(-3, -2) # (*, h, m, d)
        
        K = self.fc_k(keys)                                                # (*, n, h*d)
        K = K.reshape(*K.shape[:-1], self.num_heads, -1).transpose(-3, -2) # (*, h, n, d)
            
        # (optional) project then reshape values
        if self.embed_values:
            V = self.fc_v(values)                                              # (*, n, h*d)
            V = V.reshape(*V.shape[:-1], self.num_heads, -1).transpose(-3, -2) # (*, h, n, d) 
        else:
            V = values.unsqueeze(-3) # (*, 1, n, v)

        # compatibility "energies"
        e = self.compat(Q, K) # (*, h, m, n)

        # apply attention head-wise
        if self.act is not None:
            a = self.act(e)        # (*, h, m, n)
            O = torch.matmul(a, V) # (*, h, m, dv) where dv = d if embed_values else v
        else:
            O = torch.matmul(e, V) # (*, h, m, dv) where dv = d if embed_values else v
        
        # "concatenate" results of attention
        O = O.transpose(-3, -2)          # (*, m, h, dv)
        O = O.reshape(*O.shape[:-2], -1) # (*, m, h*dv)
        
        # (optional) project attention results back to value_size
        if self.reduce:
            return self.fc_o(O) # (*, m, v)
        else:
            return O # (*, m, h*dv)

    # convenience function used to select a particular 
    # activation function from a string argument
    def _get_attention_activation(self, activation, scalar, dim):
        if activation == 'none':
            return None
        if activation == 'softmax':
            return nn.Softmax(dim=dim)
        elif activation == 'scaled_softmax':
            return lnn.ScaledSoftmax(scalar, dim=dim)
        raise ValueError("{} is not a valid activation for attention".format(activation))

# multihead attention block
class MAB(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(MAB, self).__init__()
        self.multihead = Attention(input_size, input_size, input_size, **kwargs)
        self.layer_norm_1 = nn.LayerNorm(input_size)
        self.rFF          = nn.Linear(input_size, input_size)
        self.layer_norm_2 = nn.LayerNorm(input_size)
        
    def forward(self, X, Y):
        H = self.layer_norm_1(X + self.multihead(X, Y, Y))
        return self.layer_norm_2(H + self.rFF(H))
    
# self-attention block
class SAB(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(SAB, self).__init__()
        self.MAB = MAB(input_size, **kwargs)
        
    def forward(self, X):
        return self.MAB(X, X)