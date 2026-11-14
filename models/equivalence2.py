"""Functional equivalence of the int8 models (TFLite reference interpreter)
with the float host models used by the simulator, on 2,000 random
validation inputs per stage, for every model and seed. Decisions compared:
S1 ranking (Spearman correlation of the scores), S2 admission at theta2,
S3 prefetch at theta3, S4 arrival at theta4. Also reports the change of the
validation AUC. Usage: python3 equivalence2.py <data dir> <model dir> <theta3> <stages> <seeds>"""
import sys, json, numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import tensorflow as tf
data, md, th3, stages, seeds = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], [int(x) for x in sys.argv[5].split(',')]
TH = {'1': None, '2': 0.3, '3': th3, '4': 0.2}
def fwd(W, X):
    A = (X - W['mu'].ravel()) / W['sd'].ravel(); L = int(W['nlayers'][0, 0])
    for l in range(1, L + 1):
        A = A @ W[f'W{l}'] + W[f'b{l}'].ravel()
        if l < L: A = np.maximum(A, 0)
    return 1 / (1 + np.exp(-A[:, 0]))
def lite(path, X):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]; out = np.zeros(len(X))
    for r in range(len(X)):
        s, z = i['quantization']; x = np.clip(np.round(X[r:r+1] / s + z), -128, 127).astype(np.int8)
        it.set_tensor(i['index'], x); it.invoke(); s2, z2 = o['quantization']
        out[r] = (it.get_tensor(o['index']).astype(np.float32)[0, 0] - z2) * s2
    return out
res = {}
for j in stages:
    va = np.loadtxt(f'{data}/va_s{j}.csv', delimiter=','); ix = np.random.default_rng(0).choice(len(va), min(2000, len(va)), replace=False)
    X, y = va[ix, :-1].astype(np.float32), va[ix, -1]
    for m in ('lin', 'mlp_s', 'mlp_l'):
        for s in seeds:
            W = loadmat(f'{md}/weights/s{j}_{m}_seed{s}.mat'); f = fwd(W, X)
            q = lite(f'{md}/tflite/s{j}_{m}_seed{s}.tflite', (X - W['mu'].ravel()) / W['sd'].ravel())
            agree = float(spearmanr(f, q).correlation) if TH[j] is None else float(np.mean((f >= TH[j]) == (q >= TH[j])))
            dauc = float(roc_auc_score(y, q) - roc_auc_score(y, f)) if 0 < y.mean() < 1 else 0.0
            res[f's{j}_{m}_seed{s}'] = dict(agree=agree, dauc=dauc); print(j, m, s, round(agree, 4), round(dauc, 4), flush=True)
json.dump(res, open(f'{md}/equivalence2.json', 'w'), indent=1)
for j in stages:
    a = [v['agree'] for k, v in res.items() if k.startswith(f's{j}_')]; d = [abs(v['dauc']) for k, v in res.items() if k.startswith(f's{j}_')]
    lo = {m: min(v['agree'] for k, v in res.items() if k.startswith(f's{j}_{m}_')) for m in ('lin', 'mlp_s', 'mlp_l')}
    print(f'S{j}: agreement {min(a):.4f}..{max(a):.4f} per model min {lo}; max |dAUC| {max(d):.4f}')
