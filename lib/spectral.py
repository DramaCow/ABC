import torch
import numpy as np
from numpy.linalg import eigvalsh

def eigengaps(A):
    """
    Predict number of clusters based on the eigengap.

    Parameters
    ----------
    A : array-like, shape: (*, n, n)
        Affinity matrices. Each element of a matrix contains a measure of similarity between two data points.
    """
    
    device = A.device
        
    # assume square matrices
    n = A.shape[-1]

    # degree vector
    deg = A.sum(-1).unsqueeze(-2)                        # (*, 1, n)
    
    # inverse sqrt degree matrix:
    # since degree matrix is a diagonal matrix, the 
    # inverse is just the reciprocal of the diagonals
    D = ((1 / deg) * torch.eye(n, device=device)).sqrt() # (*, n, n)

    # normalised Laplacian matrix
    L = torch.matmul(D, torch.matmul(A, D))              # (*, n, n)

    # eigengaps defined as the difference between consecutive sorted
    # (descending) eigenvalues of the normalised Laplacian matrix
    gaps = np.ascontiguousarray(np.flip(np.diff(eigvalsh(L.cpu())), axis=-1))
    
    return torch.from_numpy(gaps) # result should be on cpu