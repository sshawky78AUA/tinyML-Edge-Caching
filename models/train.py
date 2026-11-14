"""Train, quantize (int8 PTQ) and export the candidate models of every stage.

Inputs : ../sim/data/{tr,va}_s{1..4}.csv  (features..., label), written by
         export_training.m from the synthetic trace.
Outputs: tflite/<stage>_<model>.tflite   full-integer int8 models
         weights/<stage>_<model>.mat     float weights + normalization for MATLAB
         summary.csv                     validation metric, MACs, sizes, int8/float agreement
Usage  : python3 train.py <seed>
"""
import sys, os, json
import numpy as np, tensorflow as tf
from scipy.io import savemat

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
tf.keras.utils.set_random_seed(seed)
D = os.environ.get('DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim', 'data'))
os.makedirs('tflite', exist_ok=True); os.makedirs('weights', exist_ok=True)
FE, Q3 = 512, 3

def load(part, j):
    a = np.loadtxt(os.path.join(D, f'{part}_s{j}.csv'), delimiter=',')
    return a[:, :-1].astype(np.float32), a[:, -1]

def dense_model(nin, hidden):
    x = inp = tf.keras.Input((nin,))
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inp, out)

def emb_model(d, hidden):
    inp = tf.keras.Input((Q3,), dtype='int32')
    x = tf.keras.layers.Embedding(FE + 3, d)(inp)
    x = tf.keras.layers.Flatten()(x)
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation='relu')(x)
    out = tf.keras.layers.Dense(FE + 1)(x)      # logits; FE+1 = out of vocabulary
    return tf.keras.Model(inp, out)

SPECS = {  # name: (binary hidden layers, S3 embedding dim, S3 hidden)
    'lin':   ([],        8,  []),
    'mlp_s': ([16, 16],  8,  [32]),
    'mlp_l': ([64, 64],  16, [64]),
}

def macs(model):
    n = 0
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Dense):
            n += int(np.prod(l.kernel.shape))
    return n

def to_tflite(model, rep, int_input):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    def gen():
        for i in range(min(500, len(rep))):
            yield [rep[i:i+1]]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if not int_input:
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()

def run_tflite(buf, X):
    it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for r in range(len(X)):
        x = X[r:r+1]
        if i['dtype'] == np.int8:
            s, z = i['quantization']; x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        else:
            x = x.astype(i['dtype'])
        it.set_tensor(i['index'], x); it.invoke()
        y = it.get_tensor(o['index']).astype(np.float32)
        if o['dtype'] == np.int8:
            s, z = o['quantization']; y = (y - z) * s
        out.append(y[0])
    return np.array(out)

rows = []
for j in [int(c) for c in os.environ.get('STAGES', '1234')]:
    Xtr, ytr = load('tr', j); Xva, yva = load('va', j)
    for name, (hid, d, hid3) in SPECS.items():
        key = f's{j}_{name}'
        if False:   # S3 is a dense candidate scorer (6 features), like S2
            m = emb_model(d, hid3)
            m.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m.fit(Xtr.astype(np.int32), (ytr - 1).astype(np.int32), epochs=8, batch_size=256, verbose=0)
            logit = m.predict(Xva.astype(np.int32), verbose=0)
            top = np.argsort(-logit[:, :FE], axis=1)[:, :2] + 1
            metric = float(np.mean([yva[r] in top[r] for r in range(len(yva))]))   # top-2 hit
            buf = to_tflite(m, Xtr.astype(np.int32), True)
            nchk = 300
            ref = np.argmax(logit[:nchk, :FE], 1)
            q = run_tflite(buf, Xva[:nchk].astype(np.int32))
            agree = float(np.mean(np.argmax(q[:, :FE], 1) == ref))
            mu, sd = np.zeros(Q3), np.ones(Q3)
        else:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
            m = dense_model(Xtr.shape[1], hid)
            m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(name='auc')])
            m.fit(Ztr, ytr, epochs=6, batch_size=512, verbose=0)
            p = m.predict(Zva, verbose=0)[:, 0]
            metric = float(tf.keras.metrics.AUC()(yva, p).numpy())            # ROC AUC
            buf = to_tflite(m, Ztr, False)
            nchk = 1000
            q = run_tflite(buf, Zva[:nchk])[:, 0]
            thr = {1: 0.5, 2: 0.3, 3: 0.15, 4: 0.2}[j]
            agree = float(np.mean((q >= thr) == (p[:nchk] >= thr)))
        open(f'tflite/{key}_seed{seed}.tflite', 'wb').write(buf)
        W = {'mu': mu, 'sd': sd, 'kind': 'dense'}
        li = 0
        for l in m.layers:
            if isinstance(l, tf.keras.layers.Embedding):
                W['E'] = l.get_weights()[0]
            if isinstance(l, tf.keras.layers.Dense):
                li += 1; w, b = l.get_weights(); W[f'W{li}'] = w; W[f'b{li}'] = b
        W['nlayers'] = li
        savemat(f'weights/{key}_seed{seed}.mat', W)
        nweights = int(sum(np.prod(w.shape) for w in m.get_weights()))
        rows.append(dict(key=key, seed=seed, metric=metric, macs=macs(m), params=nweights,
                         tflite_bytes=len(buf), int8_agree=agree))
        print(rows[-1], flush=True)

with open(f'summary_seed{seed}.json', 'w') as fh:
    json.dump(rows, fh, indent=1)
