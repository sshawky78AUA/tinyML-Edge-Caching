"""theta_3: the threshold with the highest F1 score of the MLP-S model of S3
(seed 1) on the validation examples, as for theta_4."""
import sys, numpy as np
from scipy.io import loadmat
data, wdir = sys.argv[1], sys.argv[2]
va = np.loadtxt(f'{data}/va_s3.csv', delimiter=','); X, y = va[:, :-1], va[:, -1]
W = loadmat(f'{wdir}/s3_mlp_s_seed1.mat'); A = (X - W['mu'].ravel()) / W['sd'].ravel()
for l in range(1, int(W['nlayers'][0, 0]) + 1):
    A = A @ W[f'W{l}'] + W[f'b{l}'].ravel(); A = np.maximum(A, 0) if l < int(W['nlayers'][0, 0]) else A
p = 1 / (1 + np.exp(-A[:, 0])); best = (0, 0)
for th in np.round(np.arange(0.05, 0.51, 0.05), 2):
    s = p >= th; tp = (s & (y == 1)).sum(); pr = tp / max(1, s.sum()); rc = tp / y.sum()
    f1 = 2 * pr * rc / max(1e-12, pr + rc); print(th, round(pr, 3), round(rc, 3), round(f1, 3)); best = max(best, (f1, th))
print('theta3', best[1])
