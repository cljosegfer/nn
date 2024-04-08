import torch
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GG:
    def __init__(self) -> None:
        pass

    def torch(self, X: np.ndarray, p: float = 2, progress = False):
        X = torch.Tensor(X).to(DEVICE)
        n = X.shape[0]
        F = torch.cdist(X, X, p = p)**p
        F.fill_diagonal_(float('inf'))

        adj = torch.zeros((n,n), dtype=torch.bool).to(DEVICE)
        if progress:
          iterador = tqdm(range(n-1))
        else:
          iterador = range(n-1)
        for i in iterador:
            A = F[i]+F[i+1:]
            idx_min = torch.argmin(A, axis=1)
            a = A[torch.arange(A.shape[0]), idx_min] - F[i, i+1:]
            adj[i, i+1:] = torch.where(a > 0, 1, 0)
        adj = adj + adj.T
        return adj.cpu()

    def divorciado(self, X_train: np.ndarray, X_test: np.ndarray, p: float = 2, progress = False):
        X_train = torch.Tensor(X_train).to(DEVICE)
        X_test = torch.Tensor(X_test).to(DEVICE)
        n = X_train.shape[0]
        N = X_test.shape[0]
        F = torch.cdist(X_train, X_train, p = p)**p
        F.fill_diagonal_(float('inf'))
        Ft = torch.cdist(X_test, X_train, p = p)**p

        adj = torch.zeros((N, n), dtype=torch.bool).to(DEVICE)
        if progress:
          iterador = tqdm(range(N))
        else:
          iterador = range(N)
        for i in iterador:
            A = Ft[i] + F
            idx_min = torch.argmin(A, axis=1)
            a = A[torch.arange(A.shape[0]), idx_min] - Ft[i]
            adj[i, :] = torch.where(a > 0, 1, 0)
        return adj.cpu()

# H = torch.load('data/H_train.pt')
# H_train, H_test = train_test_split(H, test_size = 0.1, random_state = 42)
H_train = torch.load('data/H_train.pt')
H_test = torch.load('data/H_test.pt')

# bootstrap
ps = [128, 64, 32, 16, 8, 4, 2]
for p in ps:
    print(p)
    ggclass = GG()
    # p = 5

    tol = 0.01
    eta = 0.5
    K = int(np.log(tol) / np.log(1 - eta)) + 1
    # K = 1

    N = H_train.shape[0]
    n = H_test.shape[0]
    btsz = int(N * eta)
    idx = np.arange(N)

    adjb = torch.ones((n, N), dtype=torch.bool)
    for epoch in tqdm(range(K)):
        np.random.shuffle(idx)
        for b in range(0, N, btsz):
            idx_batch = idx[b:min(b+btsz, N)]
            X_batch = H_train[idx_batch, :]
            adjb[:, idx_batch] *= ggclass.divorciado(X_batch / 4, H_test / 4, p = p, progress = False)
            # adjb[:, idx_batch] *= ggclass.divorciado(X_batch, H_test, p = p, progress = True)
    # torch.save(adjb, 'data/gg_val_bootstrap_{}.pt'.format(p))
    torch.save(adjb, 'data/gg_test_bootstrap_{}.pt'.format(p))
