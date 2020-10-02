import numpy as np
import torch
from torch.utils.data import Dataset
import itertools

from .circles import *

# uniformly-distributed nonnegative integer vector on simple hyperplane
def univosh(ndim, c, b):
    """
    returns x = [ x_1, x_2, ... , x_n ]
    satisfying the following:
    x_1 + x_2 + ... + x_n - c = 0 AND for all i, x_i <= b_i 
    """
    # assert sum(b) < c
    if sum(b) == c:
        return b
    while True:
        x = np.diff(sorted(np.concatenate(([0,], np.random.randint(0, c+1, size=ndim-1), [c,]))))
        if all(x <= b):
            return x  

def random_clusters(category, seq_length, size, k_range=None, weighted=False):
    num_classes = len(category) # num classes in category
        
    sample_limits = np.array([len(clazz) for clazz in category])
    if sum(sample_limits) < seq_length:
        raise Exception('Subsets cannot be generated: not enough samples.')
        
    # k_min = argmin_k (b_1 + ... + b_k) >= c, for b_1 <= b_2 <= ...
    # NOTE: (reverse=True will make b_1 >= b2 >= ...)
    k_min = next(idx for idx, cumsum in enumerate(itertools.accumulate(sorted(sample_limits, reverse=False))) if cumsum >= seq_length) + 1
    k_max = min(num_classes, seq_length)
    if k_range is not None:
        k_min = max(k_min, k_range[0])
        k_max = min(k_max, k_range[1])
    assert k_min <= k_max

    classes = torch.empty((size, seq_length), dtype=int)
    indices = torch.empty((size, seq_length), dtype=int)
    
    if weighted:
        weights = 1 / (np.arange(k_min, k_max+1)) ; weights /= weights.sum()
        counts  = torch.from_numpy(np.random.choice(np.arange(k_min, k_max+1), size=size, p=weights))
    else:
        counts = torch.from_numpy(np.random.randint(k_min, k_max+1, size=size))

    # randomly sample from exactly k classes
    for i, k in enumerate(int(k) for k in counts):
        ls = np.random.choice(num_classes, size=k, replace=False).astype(int) # pick which k clusters we are sampling from
        fs = univosh(k, seq_length - k, sample_limits[ls] - 1) + 1            # number of elements per cluster
        # p = np.random.permutation(seq_length) # not strictly necessary for order invariant models
        classes[i] = torch.from_numpy(np.repeat(ls, fs))#[p])
        indices[i] = torch.from_numpy(np.concatenate([np.random.choice(len(category[l]), size=f, replace=False) for f, l in zip(fs, ls)]))#[p])

    return classes, indices, counts

class Clusters(Dataset):
    def __init__(self, categories, seq_length, size, k_range=None, *, weighted=False, transform):
        super(Clusters, self).__init__()
        
        self.data = categories
        self.seq_length = seq_length
        self.size = size
        self.transform = transform

        num_categories = len(categories)
        freqs = np.array([size//num_categories + (i < size%num_categories) for i in range(num_categories)])
        
        self.categories = torch.from_numpy(np.repeat(np.arange(num_categories, dtype=int), freqs))
        self.classes, self.indices, self.counts = map(torch.cat, zip(*(random_clusters(category, seq_length, freq, k_range, weighted) for category, freq in zip(categories, freqs))))

    def pos_weight(self, *, chunk_size=1):
        num_pos = 0

        # due to memory constraints, we count the number
        # of positive samples in chunks
        for i in np.arange(0, self.size, chunk_size):
            classes = self.classes[i:i+chunk_size]
            A = (classes.unsqueeze(-2) == classes.unsqueeze(-1))
            num_pos += A.sum()

        # there may be a smaller chunk at the end
        if self.size % chunk_size > 0:
            i = (self.size // chunk_size) * chunk_size
            classes = self.classes[i:]
            A = (classes.unsqueeze(-2) == classes.unsqueeze(-1))
            num_pos += A.sum()

        num_neg = self.classes.numel() * self.seq_length - num_pos

        return num_neg.float() / num_pos.float()

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        category = self.categories[i]
        data     = self.data[category]
        classes  = self.classes[i]
        indices  = self.indices[i]
        count    = self.counts[i]

        X = torch.stack([self.transform(data[clazz][index]) for clazz, index in zip(classes, indices)])
        
        return category, X, classes, count