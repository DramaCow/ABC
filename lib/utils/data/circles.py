import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi
import torch
from torch.utils.data import Dataset

def _sample_in_unit_circle(size=1):
    radii  = np.sqrt(np.random.uniform(0, 1, size=size))
    thetas = np.random.uniform(0, 2*pi, size=size)
    return np.array([[r*cos(theta), r*sin(theta)] for r, theta in zip(radii, thetas)])

def generate_instance(L, k):
    centers = torch.from_numpy(0.5 * _sample_in_unit_circle(k)) # (k, 2)
    radii   = torch.clamp(torch.normal(0.3, 0.1, size=(k,)), min=0.2, max=0.4) # (k,)
    freqs   = np.random.multinomial(L-k, [1/k,]*k) + 1
    labels  = torch.from_numpy(np.repeat(np.arange(k), freqs)) # (L,)
    points = torch.empty((L, 2))
    cumsum = 0
    for i, freq in enumerate(freqs):
        angles = np.linspace(0, 2*pi, num=freq, endpoint=False) + np.random.uniform(0, 2*pi/freq, size=freq)
        directions = torch.tensor([[cos(angle), sin(angle)] for angle in angles])
        points[cumsum:cumsum+freq] = centers[i] + radii[i].unsqueeze(-1) * directions
        cumsum += freq
    return points, labels, (centers, radii)

class CirclesDataset(Dataset):
    def __init__(self, N, L):
        self.size = N

        self.points = torch.empty((N, L, 2))
        self.labels = torch.empty((N, L))
        self.counts = torch.empty((N,), dtype=int)

        self.centers = []
        self.radii = []

        for i, k in enumerate(torch.randint(4,5, size=(N,))):
            points, labels, (centers, radii) = generate_instance(L, int(k))

            self.points[i] = points
            self.labels[i] = labels
            self.counts[i] = k

            self.centers.append(centers)
            self.radii.append(radii)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        points   = self.points[i]
        labels   = self.labels[i]
        counts   = self.counts[i]
        centers  = self.centers[i]
        radii    = self.radii[i]
        return points, labels, counts #, (centers, radii)

    def plot(self, ax, index, plabels=None):
        points   = self.points[index]
        labels   = self.labels[index] if plabels is None else plabels
        count    = self.counts[index]
        centers  = self.centers[index]
        radii    = self.radii[index]

        for i in range(count):
            p = ax.plot(points[labels==i, 0], points[labels==i, 1], '.')
            circle = plt.Circle(centers[i], radii[i], color=p[0].get_color(), fill=False, alpha=0.0)
            ax.add_artist(circle)

if __name__=="__main__":
    from sklearn.cluster import SpectralClustering
    
    dataset = CirclesDataset(1000, 100)

    points, labels, k, (centers, radii) = dataset[0]
    plabels = SpectralClustering(k).fit(points).labels_

    fig, ax = plt.subplots()
    for i in range(k):
        # p = ax.plot(points[labels==i, 0], points[labels==i, 1], '.')
        p = ax.plot(points[plabels==i, 0], points[plabels==i, 1], '.')
        circle = plt.Circle(centers[i], radii[i], color=p[0].get_color(), fill=False, alpha=0.0)
        ax.add_artist(circle)
    plt.show()  