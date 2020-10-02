import torch

# multiplicative compatibility
def mulcomp(X, Y):
    """
    X : (*, m, H)
    Y : (*, n, H)
    out : (*, m, n) subject to broadcasting
    """
    Yt = torch.transpose(Y, -2, -1) # (*, H, n)
    return torch.matmul(X, Yt)