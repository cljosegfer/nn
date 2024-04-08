import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr

from models import kNN, sNN, sggNN

H_train = torch.load('data/H_train.pt')
y_train = torch.load('data/y_train.pt')
s_train = torch.load('data/s_train.pt')

H_test = torch.load('data/H_test.pt')
y_test = torch.load('data/y_test.pt')
s_test = torch.load('data/s_test.pt')

# # knn ------------------------------------------
# print('knn')

# # grid = [1, 2, 3, 4, 5]
# grid = np.linspace(start = 1, stop = 100, num = 60, dtype = int)
# log = []
# for param in tqdm(grid):
#     model = kNN(k = param)
#     model.fit(H_train, y_train)
#     q = model.q(H_test, y_test)
#     log.append([pearsonr(q, s_test)[0], torch.mean(q)])

# log = np.array(log)
# plt.figure()
# plt.scatter(grid, log[:, 0]);
# plt.scatter(grid, log[:, 1]);
# plt.savefig('output/knn.png')

# # snn ------------------------------------------
# print('snn')

# grid = np.logspace(start = -2, stop = 0.75, num = 20)
# # grid = [0.1, 1]
# log = []
# for param in tqdm(grid):
#     model = sNN(tau = param)
#     model.fit(H_train, y_train)
#     q = model.q(H_test, y_test)
#     log.append([pearsonr(q, s_test)[0], torch.mean(q)])

# log = np.array(log)
# plt.figure()
# plt.scatter(grid, log[:, 0]);
# plt.scatter(grid, log[:, 1]);
# plt.savefig('output/snn.png')

# sggnn ------------------------------------------
print('sggnn')

grid = np.logspace(start = -2, stop = 0.75, num = 600)
# grid = [0.1, 1]
log = []
for param in tqdm(grid):
    model = sggNN(tau = param)
    model._gg()
    model.fit(H_train, y_train)
    q = model.q(H_test, y_test)
    log.append([pearsonr(q, s_test)[0], torch.mean(q)])

log = np.array(log)
plt.figure()
plt.scatter(grid, log[:, 0]);
plt.scatter(grid, log[:, 1]);
plt.savefig('output/sggnn.png')
