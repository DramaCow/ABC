import os
import pickle
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


def load(path):
    with open(path, 'rb') as file:
        cifar100 = pickle.load(file, encoding='bytes')
    labels = np.array(cifar100[b'fine_labels']) 
    data   = np.array(cifar100[b'data']).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    return [[Image.fromarray(arr) for arr in data[labels == i]] for i in np.unique(labels)]


# taken from: https://github.com/k-han/DTC/blob/dc036e5082eeeb0d11b45c5ef768c1d0a5b0b560/data/utils.py#L20
class RandomTranslateWithReflect:
    """Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class VGG(nn.Module):
    def __init__(self, out_dim=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.last = nn.Linear(512, out_dim)

    def forward(self, x):
        dims_hi = x.shape[:-3]
        x = x.reshape(-1, 3, 32, 32)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.last(x)
        x = x.reshape(*dims_hi, -1)
        return x


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
    name = 'cifar100'

    train_transform = transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    model = nn.Sequential(VGG(out_dim=80), AffinityMul(80, 128))
    writer = SummaryWriter(log_dir=os.path.join(LOGDIR, name))
    writer.add_graph(model, (torch.Tensor(1, 100, 3, 32, 32),))
    model = model.to(device)

    # load raw data
    train_path = os.path.join('_data', 'cifar-100-python', 'train')
    test_path  = os.path.join('_data', 'cifar-100-python', 'test')

    train_data = load(train_path)
    test_data = load(test_path)

    combined_data = [train_class + test_class for train_class, test_class in zip(train_data, test_data)]

    training   = [combined_data[:90],]
    validation = [combined_data[90:],]
    evaluation = [combined_data[90:],]

    if TRAIN:
        # learning hyperparams
        lr                 = 1e-4
        batch_size         = 128
        accumulation_steps = 1 # number of batches to accumulate gradients before optim step
        val_freq           = 1000
        checkpoint_freq    = 1000

        val_set = Clusters(validation, 128, 2000, k_range=(2, 10), transform=test_transform)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        optim = torch.optim.Adam(model.parameters(), lr=lr)

        print('TRAINING...')
        optim.zero_grad()
        train_batch_loss = 0
        train_step, val_step = 0, 0
        for epoch in range(20):
            train_set = Clusters(training, 128, 500000, k_range=(2, 10), transform=train_transform)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            pos_weight = train_set.pos_weight(chunk_size=128)
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

    batch_size = 25

    eval_set    = Clusters(evaluation, 128, 10000, k_range=(2, 10), transform=test_transform)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    eigen_solver = 'arpack'
     
    model.load_state_dict(torch.load(last_checkpoint))
    model.eval()

    print('EVALUATING', last_checkpoint, '...')
    with torch.no_grad():
        ami1 = 0
        nmi1 = 0
        ari1 = 0

        ami2 = 0
        nmi2 = 0
        ari2 = 0
        
        for i, (alphabets, X, y, k) in enumerate(eval_loader, 1):
            X = X.to(device)
            Ap = torch.sigmoid(model(X))

            kp = (lib.eigengaps(Ap).argmax(-1) + 1).cpu()
            Ap = Ap.cpu().numpy()

            plabels1 = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(kp[j]), eigen_solver=eigen_solver) for j in range(batch_size)])
            plabels2 = torch.tensor([spectral_clustering(Ap[j], n_clusters=int(k[j]),  eigen_solver=eigen_solver) for j in range(batch_size)])

            ami1 += (1 / i) * (np.mean([adjusted_mutual_info_score(y[j], plabels1[j]) for j in range(batch_size)]) - ami1)
            nmi1 += (1 / i) * (np.mean([normalized_mutual_info_score(y[j], plabels1[j]) for j in range(batch_size)]) - nmi1)
            ari1 += (1 / i) * (np.mean([adjusted_rand_score(y[j], plabels1[j]) for j in range(batch_size)]) - ari1)
    
            ami2 += (1 / i) * (np.mean([adjusted_mutual_info_score(y[j], plabels2[j]) for j in range(batch_size)]) - ami2)
            nmi2 += (1 / i) * (np.mean([normalized_mutual_info_score(y[j], plabels2[j]) for j in range(batch_size)]) - nmi2)
            ari2 += (1 / i) * (np.mean([adjusted_rand_score(y[j], plabels2[j]) for j in range(batch_size)]) - ari2)

        print(ami1, nmi1, ari1, ami2, nmi2, ari2)

    print('DONE')

    writer.close()