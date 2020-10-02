import os
from os.path import join
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.metrics import adjusted_rand_score 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lib
import lib.nn as lnn
from lib.utils.data.dataset import CirclesDataset


class Pairwise(nn.Module):
    def __init__(self, input_size, hidden_size=None, writer=None):
        super(Pairwise, self).__init__()

        if hidden_size is None:
            hidden_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
        )

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)

        self.compat = lnn.AddComp(hidden_size)

    def forward(self, X, Y, k):
        X = self.fc(X)
        Q = self.fc_q(X)
        K = self.fc_k(X)
        E  = self.compat(Q, K)                   # (N, L, L)
        logits = 0.5 * (E + E.transpose(-2, -1)) # force symmetry
        return logits


class AffinityMul(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(AffinityMul, self).__init__()

        if hidden_size is None:
            hidden_size = input_size

        config = {
            'num_heads'     : 1,
            'symmetric'     : False,
            'embed_values'  : True,
            'compatibility' : 'multiplicative',
            'activation'    : 'scaled_softmax',
            'reduce'        : False,
        }

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.enc = lnn.SAB(hidden_size, **config)
        self.enc2 = lnn.SAB(hidden_size, **config)

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)

        self.compat = lnn.MulComp(hidden_size)

    def forward(self, X):
        X1 = self.fc(X)
        X2 = self.enc(X1)                        # (N, L, H)
        X2 = self.enc2(X2)
        Q  = self.fc_q(X2)
        K  = self.fc_k(X2)
        E  = self.compat(Q, K)                   # (N, L, L)
        logits = 0.5 * (E + E.transpose(-2, -1)) # force symmetry
        return logits


class AffinityAdd(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(AffinityAdd, self).__init__()

        if hidden_size is None:
            hidden_size = input_size

        config = {
            'num_heads'     : 1,
            'symmetric'     : False,
            'embed_values'  : True,
            'compatibility' : 'additive',
            'reduce'        : False,
        }

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.enc = lnn.SAB(hidden_size, **config)
        self.enc2 = lnn.SAB(hidden_size, **config)

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)

        self.compat = lnn.AddComp(hidden_size)

    def forward(self, X):
        X1 = self.fc(X)
        X2 = self.enc(X1)                        # (N, L, H)
        X2 = self.enc2(X2)
        Q  = self.fc_q(X2)
        K  = self.fc_k(X2)
        E  = self.compat(Q, K)                   # (N, L, L)
        logits = 0.5 * (E + E.transpose(-2, -1)) # force symmetry
        return logits


# ============
# === MAIN ===
# ============


if __name__=="__main__":
    LOGDIR        = '_logs'
    CHECKPOINTDIR = '_checkpoints'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = 'circles1'

    model  = AffinityAdd(2, 128)
    writer = SummaryWriter(log_dir=os.path.join(LOGDIR, name))
    writer.add_graph(model, (torch.Tensor(1, 100, 1, 2),))
    model = model.to(device)

    # learning hyperparams
    lr                 = 1e-3
    batch_size         = 128
    accumulation_steps = 1 # number of batches to accumulate gradients before optim step
    val_freq           = 1000
    checkpoint_freq    = 1000

    train_set = CirclesDataset(1000000, 20)
    val_set   = CirclesDataset(400, 20)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    pos_weight = torch.tensor(1)#train_set.pos_weight(chunk_size=128)
    print('pos_weight =', pos_weight)

    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn = loss_fn.to(device)

    # ================
    # === TRAINING ===
    # ================

    print('TRAINING...')
    optim.zero_grad()
    train_batch_loss = 0
    train_step, val_step = 0, 0
    for batch_id, (X, y, _) in enumerate(train_loader, 1):
        X = X.to(device)
        A = (y.unsqueeze(-2) == y.unsqueeze(-1)).float().to(device)

        Ep = model(X)
        loss = loss_fn(Ep, A) / accumulation_steps
        loss.backward()

        train_batch_loss += loss.item()

        # optimize
        if batch_id % accumulation_steps == 0:
            optim.step()
            optim.zero_grad()
            writer.add_scalar('Training/Loss', train_batch_loss, train_step)
            train_batch_loss = 0    
            train_step += 1

        # validation
        if batch_id % val_freq == 0:
            model.eval()
            with torch.no_grad():
                ari = 0
                for i, (X, y, k) in enumerate(val_loader, 1):
                    X = X.to(device)
                    Ap = torch.sigmoid(model(X))
                    kp = (lib.eigengaps(Ap).argmax(-1) + 1).cpu()
                    Ap = Ap.cpu()

                    plabels = torch.tensor([SpectralClustering(int(kp[j]), affinity='precomputed').fit(Ap[j]).labels_ for j in range(batch_size)])

                    ari = ari + (1 / i) * (np.mean([adjusted_rand_score(y[j], plabels[j]) for j in range(batch_size)]) - ari)
                writer.add_scalar('Training/ValidationARI', ari, val_step)
                val_step += 1
            model.train()

        # checkpoint
        if batch_id % checkpoint_freq == 0:
            path = join(CHECKPOINTDIR, '%s_%09d.ckpt' % (name, batch_id))
            torch.save(model.state_dict(), path)
    print('DONE')

    # ==================
    # === EVALUATION ===
    # ==================

    batch_size = 50

    eval_set    = CirclesDataset(1000, 20)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    eigen_solver = 'arpack'

    model.eval()

    print('EVALUATING...')
    with torch.no_grad():
        ari = 0
        for i, (X, y, k) in enumerate(eval_loader, 1):
            X = X.to(device)
            Ap = torch.sigmoid(model(X))

            kp = (lib.eigengaps(Ap).argmax(-1) + 1).cpu()
            Ap = Ap.cpu().numpy()

            plabels = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(kp[j]), eigen_solver=eigen_solver) for j in range(batch_size)])

            ari = ari + (1 / i) * (np.mean([adjusted_rand_score(y[j], plabels[j]) for j in range(batch_size)]) - ari)
        print('ari =', ari)
    print('DONE')

    writer.close()