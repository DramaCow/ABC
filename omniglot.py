import os
from os.path import basename, normpath, join
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


def load_alphabets(path):            
    return { basename(normpath(apath)) : [[Image.open(filename).convert('L') for filename in iglob(join(cpath, '*'))] for cpath in iglob(join(apath, '*'))] for apath in iglob(join(path, '*')) }


class ImageEncoder28(nn.Module):
    def __init__(self):
        super(ImageEncoder28, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, X):
        dims_hi = X.shape[:-3]
        X = X.reshape(-1, 1, 28, 28) # (*, 1, 28, 28)
        X = self.conv(X)             # (*, 64, 1, 1)
        X = X.reshape(*dims_hi, -1)  # (N, *, 64)
        return X   
        

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


# ============
# === MAIN ===
# ============


if __name__=="__main__":
    LOGDIR        = '_logs'
    CHECKPOINTDIR = '_checkpoints'
    TRAIN         = True  
    TEST          = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = 'run1'

    train_transform = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()

    model = nn.Sequential(ImageEncoder28(), AffinityAdd(64, 128))
    writer = SummaryWriter(log_dir=os.path.join(LOGDIR, name))
    writer.add_graph(model, (torch.Tensor(1, 100, 1, 28, 28),))
    model = model.to(device)

    # load raw data
    background = load_alphabets(join(os.getcwd(), '_data', 'images_background_small_test'))
    evaluation = load_alphabets(join(os.getcwd(), '_data', 'images_evaluation_small_test'))

    background_names = list(background.keys())
    evaluation_names = list(evaluation.keys())

    if not TEST:
        from random import shuffle
        shuffle(background_names)
        training_names   = background_names[:26]
        validation_names = background_names[26:]
    else:
        training_names   = background_names
        validation_names = evaluation_names

    print('training alphabets:',   training_names)
    print('validation alphabets:', validation_names)
    print('evaluation alphabets:', evaluation_names)

    if not TEST:
        training   = [background[key] for key in training_names]
        validation = [background[key] for key in validation_names]
    else:
        training   = list(background.values())
        validation = list(evaluation.values())
        evaluation = list(evaluation.values())

    if TRAIN:
        # learning hyperparams
        lr                 = 1e-4
        batch_size         = 42
        accumulation_steps = 3 # number of batches to accumulate gradients before optim step
        val_freq           = 1000
        checkpoint_freq    = 1000

        train_set = Clusters(training, 100, 10000000, k_range=(5, 35), transform=train_transform)
        val_set   = Clusters(validation, 100, 400, k_range=(5, 35), transform=test_transform)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        pos_weight = train_set.pos_weight(chunk_size=128)
        print('pos_weight =', pos_weight)

        optim   = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_fn = loss_fn.to(device)

        print('TRAINING...')
        optim.zero_grad()
        train_batch_loss = 0
        train_step, val_step = 0, 0
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
                path = join(CHECKPOINTDIR, '%s_%09d.ckpt' % (name, batch_id))
                torch.save(model.state_dict(), path)
        print('DONE')

    batch_size = 50
    num_alphabets = len(evaluation)

    eval_set    = Clusters(evaluation, 100, 20000, k_range=(5, 47), transform=test_transform)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    eigen_solver = 'arpack'

    checkpoints = (
        'run1_000238000.ckpt',
    )

    for ckpt in checkpoints:        
        model.load_state_dict(torch.load(join(CHECKPOINTDIR, ckpt)))
        model.eval()

        print('EVALUATING', ckpt, '...')
        with torch.no_grad():
            counts = np.zeros(num_alphabets)

            ami1 = np.zeros(num_alphabets)
            nmi1 = np.zeros(num_alphabets)
            ari1 = np.zeros(num_alphabets)

            ami2 = np.zeros(num_alphabets)
            nmi2 = np.zeros(num_alphabets)
            ari2 = np.zeros(num_alphabets)

            errors = [[] for _ in range(num_alphabets)]
            
            for i, (alphabets, X, y, k) in enumerate(eval_loader, 1):
                X = X.to(device)
                Ap = torch.sigmoid(model(X))

                kp = (lib.eigengaps(Ap).argmax(-1) + 1).cpu()
                Ap = Ap.cpu().numpy()

                plabels1 = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(kp[j]), eigen_solver=eigen_solver) for j in range(batch_size)])
                plabels2 = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(k[j]),  eigen_solver=eigen_solver) for j in range(batch_size)])

                for a in range(num_alphabets):
                    m = (alphabets == a).sum()
                    if m == 0:
                        continue
                    counts[a] += m

                    ami1[a] += (1 / counts[a]) * (sum([adjusted_mutual_info_score(y[j], plabels1[j]) for j in range(batch_size) if alphabets[j] == a]) - m * ami1[a])
                    nmi1[a] += (1 / counts[a]) * (sum([normalized_mutual_info_score(y[j], plabels1[j]) for j in range(batch_size) if alphabets[j] == a]) - m * nmi1[a])
                    ari1[a] += (1 / counts[a]) * (sum([adjusted_rand_score(y[j], plabels1[j]) for j in range(batch_size) if alphabets[j] == a]) - m * ari1[a])
                    
                    ami2[a] += (1 / counts[a]) * (sum([adjusted_mutual_info_score(y[j], plabels2[j]) for j in range(batch_size) if alphabets[j] == a]) - m * ami2[a])
                    nmi2[a] += (1 / counts[a]) * (sum([normalized_mutual_info_score(y[j], plabels2[j]) for j in range(batch_size) if alphabets[j] == a]) - m * nmi2[a])
                    ari2[a] += (1 / counts[a]) * (sum([adjusted_rand_score(y[j], plabels2[j]) for j in range(batch_size) if alphabets[j] == a]) - m * ari2[a])
            
                    errors[a].extend(kp[alphabets==a] - k[alphabets==a])

            with open(ckpt+".txt", "w") as file:
                for name, score1, score2 in zip(evaluation_names, nmi1, nmi2):
                    file.write('%s & %.4f & %.4f & - \\\\\n' % (name, score1, score2))
                file.write('\\hline\n')
                file.write('\\textbf{mean} & %.4f & %.4f & - \\\\\n' % (np.mean(nmi1), np.mean(nmi2)))
            
            with open(ckpt+"_nclusters.txt", "w") as file:
                for name, mean, std in zip(evaluation_names, [np.mean(err) for err in errors], [np.std(np.abs(err)) for err in errors]):
                    file.write('%s & %.4f & %.4f \\\\\n' % (name, mean, std))
                file.write('\\hline\n')
                file.write('\\textbf{mean} & %.4f & %.4f \\\\\n' % (np.mean(np.abs(errors)), np.std(np.abs(errors))))

        print('DONE')

    writer.close()