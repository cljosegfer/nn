
import torch
import numpy as np

from tqdm import tqdm

class NN():
  def __init__(self):
    pass

  def fit(self, X, y):
    self.X = X
    self.y = y

  def _predict(self, X, mode, y = None):
    pass

  def q(self, X, y):
    return self._predict(X = X, mode = 'q', y = y)

  def q_mean(self, X, y):
    yhat = self.q(X, y)
    return torch.mean(yhat)

class kNN(NN):
  def __init__(self, k):
    super().__init__()
    self.k = k

  def _predict(self, X, mode, y = None):
    n = X.shape[0]
    yhat = torch.zeros(n)
    delta = torch.cdist(self.X, X)
    for i in tqdm(range(n)):
      vizinhos = torch.argsort(delta[:, i], axis = 0)[:self.k]
      possible, count = torch.unique(self.y[vizinhos], return_counts = True)
      freq = count / len(vizinhos)
      if mode == 'q':
        try:
            yhat[i] = freq[y[i] == possible]
        except:
            pass

    return yhat

class sNN(NN):
  def __init__(self, tau):
    super().__init__()
    self.tau = tau

  def _predict(self, X, mode, y = None):
    n = X.shape[0]
    yhat = torch.zeros(n)
    delta = torch.cdist(self.X, X)
    for i in tqdm(range(n)):
      weights = self._softmin(delta[:, i])
      if mode == 'q':
        yhat[i] = torch.sum(weights[y[i] == self.y])

    return yhat

  def _softmin(self, x):
    return torch.exp(-self.tau * x) / torch.sum(torch.exp(-self.tau * x))

class ggNN(NN):
  def __init__(self):
    super().__init__()
    self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def fit(self, X, y):
    self.X = X
    self.y = y
    self.N = self.X.shape[0]

  def _gg(self, delta, btsz):
    val_min = torch.ones(self.N).to(self.DEVICE) * 1e6
    for b in range(0, self.N, btsz):
      X_batch = self.X[b:b+btsz, :].to(self.DEVICE)
      F_batch = torch.cdist(X_batch, self.X.to(self.DEVICE))**2
      A_batch = delta[:btsz].T + F_batch.T
      val_min_batch, _ = torch.min(A_batch, axis = 1)
      val_min, _ = torch.min(torch.stack((val_min, val_min_batch), dim = 1), dim = 1)
      del X_batch, F_batch, A_batch, val_min_batch
    a = val_min - delta
    return torch.where(a > 0, True, False).cpu()

  def _predict(self, X, mode, y = None):
    n = X.shape[0]
    yhat = torch.zeros(n)
    delta = torch.cdist(self.X, X).to(self.DEVICE)
    for i in tqdm(range(n)):
      vizinhos = self._gg(delta[:, i], btsz = self.N // 4)
      possible, count = torch.unique(self.y[vizinhos], return_counts = True)
      freq = count / torch.sum(vizinhos)
      if mode == 'q':
        try:
            yhat[i] = freq[y[i] == possible]
        except:
            pass

    return yhat
