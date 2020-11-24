import os
import pickle
from collections import OrderedDict
import numpy as np
from glob import iglob
from PIL import Image
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import lib
import lib.nn as lnn
from lib.utils.data.dataset import Clusters


def load_raw_data(filename):
    with open(filename, 'rb') as file:
        val = pickle.load(file, encoding='latin1')
    return val


def process_raw_data(data):
    # no sanity checks done here - trusting the data providers
    all_classes = OrderedDict()
    for index, key in enumerate(data['keys']):
        _, label, _ = key.split('-')
        all_classes.setdefault(label, []).append(index)
    return { label: torch.from_numpy(data['embeddings'][indices]) for label, indices in all_classes.items() }


class AffinityMul(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(AffinityMul, self).__init__()

        if hidden_size is None:
            hidden_size = input_size

        config = {
            'num_heads'     : 4,
            'symmetric'     : False,
            'embed_values'  : True,
            'compatibility' : 'multiplicative',
            'activation'    : 'scaled_softmax',
            'reduce'        : True,
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
            'num_heads'     : 4,
            'symmetric'     : False,
            'embed_values'  : True,
            'compatibility' : 'additive',
            'activation'    : 'scaled_softmax',
            'reduce'        : True,
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


if __name__=="__main__":
    LOGDIR        = '_logs'
    CHECKPOINTDIR = '_checkpoints'
    TRAIN         = True  
    TEST          = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = 'image_net_embedded_5'

    model = AffinityMul(640, 128)
    writer = SummaryWriter(log_dir=os.path.join(LOGDIR, name))
    writer.add_graph(model, (torch.Tensor(1, 100, 640),))
    model = model.to(device)

    # load raw data
    tiered_train_path = os.path.join('_data', 'embeddings', 'tieredImageNet', 'center', 'train_embeddings.pkl')
    mini_train_path = os.path.join('_data', 'embeddings', 'miniImageNet', 'center', 'train_embeddings.pkl')
    tiered_val_path = os.path.join('_data', 'embeddings', 'tieredImageNet', 'center', 'val_embeddings.pkl')
    mini_val_path = os.path.join('_data', 'embeddings', 'miniImageNet', 'center', 'val_embeddings.pkl')
    tiered_test_path = os.path.join('_data', 'embeddings', 'tieredImageNet', 'center', 'test_embeddings.pkl')
    mini_test_path = os.path.join('_data', 'embeddings', 'miniImageNet', 'center', 'test_embeddings.pkl')

    tiered_train_data = load_raw_data(tiered_train_path)
    mini_train_data = load_raw_data(mini_train_path)
    tiered_val_data = load_raw_data(tiered_val_path)
    mini_val_data = load_raw_data(mini_val_path)
    tiered_test_data = load_raw_data(tiered_test_path)
    mini_test_data = load_raw_data(mini_test_path)

    train_data = process_raw_data({
        'keys'      : np.concatenate((tiered_train_data['keys'], mini_train_data['keys'], tiered_val_data['keys'], mini_val_data['keys'])),
        'embeddings': np.concatenate((tiered_train_data['embeddings'], mini_train_data['embeddings'], tiered_val_data['embeddings'], mini_val_data['embeddings'])),
        'labels'    : np.concatenate((tiered_train_data['labels'], mini_train_data['labels'], tiered_val_data['labels'], mini_val_data['labels'])),
    })

    test_data = process_raw_data({
        'keys' : np.concatenate((tiered_test_data['keys'], mini_test_data['keys'])),
        'embeddings' : np.concatenate((tiered_test_data['embeddings'], mini_test_data['embeddings'])),
        'labels' : np.concatenate((tiered_test_data['labels'], mini_test_data['labels'])),
    })

    training   = [list(train_data.values())]
    validation = [list(test_data.values())]
    evaluation = [list(test_data.values())]

    if TRAIN:
        # learning hyperparams
        lr                 = 1e-3
        batch_size         = 128
        accumulation_steps = 1 # number of batches to accumulate gradients before optim step
        val_freq           = 1000
        checkpoint_freq    = 1000

        val_set   = Clusters(validation, 300, 2000, k_range=(2, 12), transform=lambda x: x)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        
        optim   = torch.optim.Adam(model.parameters(), lr=lr)

        print('TRAINING...')
        optim.zero_grad()
        train_batch_loss = 0
        train_step, val_step = 0, 0
        for epoch in range(10):
            train_set = Clusters(training, 300, 500000, k_range=(2, 12), transform=lambda x: x)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            
            pos_weight = train_set.pos_weight(chunk_size=128)
            print('pos_weight =', pos_weight)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_fn = loss_fn.to(device)

            for batch_id, (_, X, y, _) in enumerate(train_loader, 1):
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
                        nmi = 0
                        nmi_known = 0
                        for i, (_, X, y, k) in enumerate(val_loader, 1):
                            X = X.to(device)
                            Ap = torch.sigmoid(model(X))
                            kp = (lib.eigengaps(Ap).argmax(-1) + 1).cpu()
                            Ap = Ap.cpu()

                            # plabels = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(kp[j]), eigen_solver=eigen_solver) for j in range(batch_size)])
                            plabels = torch.tensor([SpectralClustering(int(kp[j]), affinity='precomputed').fit(Ap[j]).labels_ for j in range(batch_size)])
                            plabels_known = torch.tensor([SpectralClustering(int(k[j]), affinity='precomputed').fit(Ap[j]).labels_ for j in range(batch_size)])

                            nmi = nmi + (1 / i) * (np.mean([normalized_mutual_info_score(y[j], plabels[j]) for j in range(batch_size)]) - nmi)
                            nmi_known = nmi_known + (1 / i) * (np.mean([normalized_mutual_info_score(y[j], plabels_known[j]) for j in range(batch_size)]) - nmi_known)
                        writer.add_scalar('Training/ValidationNMI', nmi, val_step)
                        writer.add_scalar('Training/ValidationNMI_Known', nmi_known, val_step)
                        val_step += 1
                    model.train()

                # checkpoint
                if batch_id % checkpoint_freq == 0:
                    print('CHECKPOINT')
                    last_checkpoint = os.path.join(CHECKPOINTDIR, '%s_%d_%09d.ckpt' % (name, epoch, batch_id))
                    torch.save(model.state_dict(), last_checkpoint)
        print('DONE')

    # === evaluation ===

    batch_size = 50

    eval_set    = Clusters(evaluation, 300, 10000, k_range=(2, 12), transform=lambda x: x)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    eigen_solver = 'arpack'
     
    model.load_state_dict(torch.load(last_checkpoint))
    model.eval()

    print('EVALUATING', last_checkpoint, '...')
    with torch.no_grad():
        # ami1 = 0
        nmi1 = 0
        # ari1 = 0

        # ami2 = 0
        nmi2 = 0
        # ari2 = 0

        # errors = []
        
        for i, (alphabets, X, y, k) in enumerate(eval_loader, 1):
            X = X.to(device)
            Ap = torch.sigmoid(model(X))

            kp = (lib.eigengaps(Ap).argmax(-1) + 1).cpu()
            Ap = Ap.cpu().numpy()

            plabels1 = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(kp[j]), eigen_solver=eigen_solver) for j in range(batch_size)])
            plabels2 = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(k[j]),  eigen_solver=eigen_solver) for j in range(batch_size)])

            # ami1 += (1 / i) * (np.mean([adjusted_mutual_info_score(y[j], plabels1[j]) for j in range(batch_size)]) - ami1)
            nmi1 += (1 / i) * (np.mean([normalized_mutual_info_score(y[j], plabels1[j]) for j in range(batch_size)]) - nmi1)
            # ari1 += (1 / i) * (np.mean([adjusted_rand_score(y[j], plabels1[j]) for j in range(batch_size)]) - ari1)
    
            # ami2 += (1 / i) * (np.mean([adjusted_mutual_info_score(y[j], plabels2[j]) for j in range(batch_size)]) - ami2)
            nmi2 += (1 / i) * (np.mean([normalized_mutual_info_score(y[j], plabels2[j]) for j in range(batch_size)]) - nmi2)
            # ari2 += (1 / i) * (np.mean([adjusted_rand_score(y[j], plabels2[j]) for j in range(batch_size)]) - ari2)
            
            # errors.extend(np.abs(kp - k))

        # print(ami1, nmi1, ari1, ami2, nmi2, ari2)
        print('nmi (unknown) =', nmi1, 'nmi (known) =', nmi2)

    print('DONE')

    writer.close()
