
import torch
import numpy as np

from tqdm import tqdm

class NN():
  def __init__(self):
    self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def fit(self, X, y):
    self.X = X
    self.y = y

  def _predict(self, X, mode, y = None, progress = False):
    pass

  def q(self, X, y, progress = False):
    return self._predict(X = X, mode = 'q', y = y, progress = progress)
  
  def clf(self, X, progress = False):
    return self._predict(X = X, mode = 'clf', progress = progress)

class kNN(NN):
  def __init__(self, k):
    super().__init__()
    self.k = k

  def _predict(self, X, mode, y = None, progress = True):
    n = X.shape[0]
    yhat = torch.zeros(n)
    # if mode == 'clf':
    #   shat = torch.zeros(n)
    # delta = torch.cdist(self.X, X)
    delta = torch.cdist(self.X, X).to(self.DEVICE)
    if progress:
       iterador = tqdm(range(n))
    else:
       iterador = range(n)
    for i in iterador:
      vizinhos = torch.argsort(delta[:, i], axis = 0)[:self.k]
      possible, count = torch.unique(self.y[vizinhos], return_counts = True)
      freq = count / len(vizinhos)
      if mode == 'clf':
         yhat[i] = possible[np.argmax(count)]
        #  shat[i] = torch.max(freq)
      if mode == 'q':
        try:
            yhat[i] = freq[y[i] == possible]
        except:
            pass
    
    # if mode == 'clf':
    #   return yhat, shat
    return yhat

class sNN(NN):
  def __init__(self, tau):
    super().__init__()
    self.tau = tau

  def _predict(self, X, mode, y = None, progress = True):
    n = X.shape[0]
    yhat = torch.zeros(n)
    # if mode == 'clf':
    #   shat = torch.zeros(n)
    # delta = torch.cdist(self.X, X)
    delta = torch.cdist(self.X, X).to(self.DEVICE)
    if progress:
       iterador = tqdm(range(n))
    else:
       iterador = range(n)
    for i in iterador:
      weights = self._softmin(delta[:, i])
      if mode == 'q':
        yhat[i] = torch.sum(weights[y[i] == self.y])
      if mode == 'clf':
        possible = torch.unique(self.y)
        yhat[i] = possible[torch.argmax(torch.tensor([torch.sum(weights[self.y == p]) for p in possible]))]
        # shat[i] = torch.max(torch.tensor([torch.sum(weights[self.y == p]) for p in possible]))

    # if mode == 'clf':
    #   return yhat, shat
    return yhat

  def _softmin(self, x):
    return torch.exp(-self.tau * x) / torch.sum(torch.exp(-self.tau * x))

class ggNN(NN):
  def __init__(self):
    super().__init__()

  def fit(self, X, y):
    self.X = X
    self.y = y
    self.N = self.X.shape[0]

  def _gg(self, path = 'data/gg_test_bootstrap.pt'):
    self.gg = torch.load(path)

  def _predict(self, X, mode, y = None, progress = True):
    n = X.shape[0]
    yhat = torch.zeros(n)
    # if mode == 'clf':
    #   shat = torch.zeros(n)
    # delta = torch.cdist(self.X, X).to(self.DEVICE)
    if progress:
       iterador = tqdm(range(n))
    else:
       iterador = range(n)
    for i in iterador:
      # vizinhos = self._gg(delta[:, i], btsz = self.N // 4)
      vizinhos = self.gg[i, :]
      if not vizinhos.any():
        continue
      possible, count = torch.unique(self.y[vizinhos], return_counts = True)
      freq = count / torch.sum(vizinhos)
      if mode == 'clf':
         yhat[i] = possible[np.argmax(count)]
        #  shat[i] = torch.max(freq)
      if mode == 'q':
        try:
            yhat[i] = freq[y[i] == possible]
        except:
            pass

    # if mode == 'clf':
    #   return yhat, shat
    return yhat

class sggNN(ggNN):
  def __init__(self, tau):
    super().__init__()
    self.tau = tau

  def _predict(self, X, mode, y = None, progress = True):
    n = X.shape[0]
    yhat = torch.zeros(n)
    # if mode == 'clf':
    #   shat = torch.zeros(n)
    delta = torch.cdist(self.X, X).to(self.DEVICE)
    if progress:
       iterador = tqdm(range(n))
    else:
       iterador = range(n)
    for i in iterador:
      # vizinhos = self._gg(delta[:, i], btsz = self.N // 4)
      vizinhos = self.gg[i, :]
      if not vizinhos.any():
        continue
      weights = self._softmin(delta[vizinhos, i])
      if mode == 'q':
        yhat[i] = torch.sum(weights[y[i] == self.y[vizinhos]])
      if mode == 'clf':
        possible = torch.unique(self.y[vizinhos])
        yhat[i] = possible[torch.argmax(torch.tensor([torch.sum(weights[self.y[vizinhos] == p]) for p in possible]))]
        # shat[i] = torch.max(torch.tensor([torch.sum(weights[self.y[vizinhos] == p]) for p in possible]))

    # if mode == 'clf':
    #   return yhat, shat
    return yhat

  def _softmin(self, x):
    return torch.exp(-self.tau * x) / torch.sum(torch.exp(-self.tau * x))
