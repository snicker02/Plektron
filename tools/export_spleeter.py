"""Export Deezer Spleeter (MIT) 4-stem U-Net weights to Plektron's compact format.

usage: python3 export_spleeter.py <checkpoint_dir> <out_prefix> [--f32] [--i8-min N]
Writes <out_prefix>.json (manifest) and <out_prefix>-0.bin, -1.bin, … (each < 20 MB).
Large conv/deconv kernels are int8 with one float32 scale per (kh, kw, out) tap (symmetric),
smaller ones float16;
biases and batch-norm parameters stay float32. BN is pre-folded into scale/shift:
y = x * g/sqrt(var+eps) + (beta - mean*g/sqrt(var+eps)).
"""
import sys, os, json, numpy as np, tensorflow as tf

ckpt, out = sys.argv[1], sys.argv[2]
f32 = '--f32' in sys.argv
I8_MIN = int(sys.argv[sys.argv.index('--i8-min') + 1]) if '--i8-min' in sys.argv else 600000  # kernels at least this big go int8, smaller go float16
r = tf.train.load_checkpoint(ckpt)
names = {n for n, _ in tf.train.list_variables(ckpt)}
stems = ['vocals', 'drums', 'bass', 'other']
EPS = 1e-3
blob = bytearray(); layers = []

def put(arr, dtype):
    global blob
    while len(blob) % 4: blob += b'\0'
    off = len(blob); blob += arr.astype(dtype).tobytes(); return off

def kernel(name, outAxis):
    w = r.get_tensor(name + '/kernel').astype(np.float32)
    b = r.get_tensor(name + '/bias').astype(np.float32)
    e = {'name': name, 'shape': list(w.shape), 'bias': put(b, '<f4')}
    if f32:
        e['dtype'] = 'f32'; e['w'] = put(w, '<f4')
    elif w.size < I8_MIN:
        e['dtype'] = 'f16'; e['w'] = put(w, '<f2')
    else:
        # one scale per (kh, kw, out) tap: the input-channel vector of each tap shares a scale
        inAxis = 2 if outAxis == 3 else 3
        s = np.abs(w).max(axis=inAxis, keepdims=True) / 127.0
        s[s == 0] = 1.0
        q = np.clip(np.round(w / s), -127, 127)
        e['dtype'] = 'i8'; e['w'] = put(q, '<i1'); e['scale'] = put(np.squeeze(s, inAxis), '<f4'); e['outAxis'] = outAxis
    return e

def bn(i):
    p = f'batch_normalization{"" if i == 0 else "_" + str(i)}'
    g, be, mu, var = (r.get_tensor(f'{p}/{k}').astype(np.float64) for k in ('gamma', 'beta', 'moving_mean', 'moving_variance'))
    sc = g / np.sqrt(var + EPS); sh = be - mu * sc
    return {'name': p, 'n': len(g), 'scale': put(sc, '<f4'), 'shift': put(sh, '<f4')}

cname = lambda i: 'conv2d' if i == 0 else f'conv2d_{i}'
tname = lambda i: 'conv2d_transpose' if i == 0 else f'conv2d_transpose_{i}'
model = {'format': 'plektron-unet-1', 'stems': [], 'activation': None}
for si, stem in enumerate(stems):
    enc = [kernel(cname(7 * si + i), 3) for i in range(6)]
    dec = [kernel(tname(6 * si + j), 2) for j in range(6)]       # [kh,kw,out,in]
    final = kernel(cname(7 * si + 6), 3)
    bns = [bn(12 * si + k) for k in range(12)]
    model['stems'].append({'name': stem, 'enc': enc, 'dec': dec, 'final': final, 'bn': bns})
model['bytes'] = len(blob)
# split into parts < 20 MB so every file fits GitHub's 25 MB web-upload limit
PART = 20_000_000
base = os.path.basename(out)
model['parts'] = []
for i in range(0, len(blob), PART):
    name = f'{base}-{i // PART}.bin'
    open(os.path.join(os.path.dirname(out) or '.', name), 'wb').write(blob[i:i + PART])
    model['parts'].append({'file': name, 'bytes': len(blob[i:i + PART])})
json.dump(model, open(out + '.json', 'w'))
print('wrote', out, len(blob) / 1e6, 'MB in', len(model['parts']), 'parts')
