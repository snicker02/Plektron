// Spleeter 4-stem U-Net (Deezer, MIT licence) forward pass, dependency-free.
// Weights come from tools/export_spleeter.py: big kernels int8 with one scale per (kh, kw, out) tap,
// small kernels float16, biases float32, batch-norm pre-folded into scale/shift.
// Uses the WebAssembly SIMD kernel when available, plain JS otherwise.
import { conv2d } from './graph.js';
import { createSimd } from './simd.js';

const T = (shape, data) => ({ shape, data, dtype: 'float32' });

function half(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}

export class UNet4 {
  /** @param {object} manifest parsed spleeter4.json  @param {ArrayBuffer} bin spleeter4.bin */
  constructor(manifest, bin, { simd = true } = {}) {
    this.m = manifest; this.bin = bin;
    this.stems = manifest.stems.map((s) => s.name);
    this.scratch = new Float32Array(0);
    this.simd = simd ? createSimd() : null;
  }
  f32(off, n) { return new Float32Array(this.bin, off, n); }

  /** Dequantized kernel in its stored layout (conv: [KH,KW,CI,CO], convT: [KH,KW,CO,CI]). */
  kernel(e) {
    const n = e.shape.reduce((a, b) => a * b, 1);
    if (e.dtype === 'f32') return this.f32(e.w, n);
    if (e.dtype === 'f16') {
      if (!e._f) { const h = new Uint16Array(this.bin, e.w, n), f = new Float32Array(n); for (let i = 0; i < n; i++) f[i] = half(h[i]); e._f = f; }
      return e._f;
    }
    if (this.scratch.length < n) this.scratch = new Float32Array(n);
    const w = this.scratch.subarray(0, n), q = new Int8Array(this.bin, e.w, n);
    const [KH, KW, A, B] = e.shape, taps = KH * KW;
    if (e.outAxis === 3) { // conv: [tap][ci][co], scale [tap][co]
      const CI = A, CO = B, sc = this.f32(e.scale, taps * CO);
      for (let t = 0, i = 0; t < taps; t++) for (let ci = 0; ci < CI; ci++) { const so = t * CO; for (let co = 0; co < CO; co++, i++) w[i] = q[i] * sc[so + co]; }
    } else {               // convT: [tap][co][ci], scale [tap][co]
      const CO = A, CI = B, sc = this.f32(e.scale, taps * CO);
      for (let g = 0, i = 0; g < taps * CO; g++) { const v = sc[g]; for (let ci = 0; ci < CI; ci++, i++) w[i] = q[i] * v; }
    }
    return w;
  }

  conv(x, e, strides, dil) {
    const w = T(e.shape, this.kernel(e)), b = this.f32(e.bias, e.shape[3]);
    if (this.simd && dil === 1 && e.shape[3] % 16 === 0) return this.simd.conv2d(x, w, strides, 'SAME', b);
    return conv2d(x, w, strides, 'SAME', [1, dil, dil, 1], b, false);
  }

  convT(x, e) {
    const [KH, KW, CO, CI] = e.shape, src = this.kernel(e), b = this.f32(e.bias, CO);
    if (this.simd && CO % 16 === 0) {
      // reorder [tap][co][ci] → [tap][ci][co] for the vec×mat kernel
      const taps = KH * KW, n = taps * CO * CI;
      if (!this.perm || this.perm.length < n) this.perm = new Float32Array(n);
      const p = this.perm.subarray(0, n);
      for (let t = 0; t < taps; t++) { const o = t * CO * CI; for (let co = 0; co < CO; co++) for (let ci = 0; ci < CI; ci++) p[o + ci * CO + co] = src[o + co * CI + ci]; }
      return this.simd.convT(x, T([KH, KW, CI, CO], p), b);
    }
    return convTranspose2d(x, T(e.shape, src), b);
  }

  /**
   * @param {Float32Array} X magnitudes [H frames, W bins, 2 channels]
   * @returns {Object<string, Float32Array>} per-stem magnitude estimates, same layout as X
   */
  run(X, H = 512, W = 1024) {
    const out = {};
    for (const s of this.m.stems) out[s.name] = this.stem(s, X, H, W);
    return out;
  }

  stem(s, X, H, W) {
    let x = T([1, H, W, 2], X);
    const skips = [];
    // encoder: conv(5x5, stride 2) → [skip = conv output] → BN → ELU
    for (let i = 0; i < 6; i++) {
      const c = this.conv(x, s.enc[i], [1, 2, 2, 1], 1);
      skips.push(c);
      if (i < 5) { x = T(c.shape.slice(), c.data.slice()); bnInPlace(x, s.bn[i], this); elu(x.data); }
    }
    // decoder: convT(5x5, stride 2) → ELU → BN → concat(skip, ·)
    x = skips[5];
    for (let j = 0; j < 6; j++) {
      const u = this.convT(x, s.dec[j]);
      elu(u.data);
      bnInPlace(u, s.bn[6 + j], this);
      x = j < 5 ? concatC(skips[4 - j], u) : u;
    }
    // final: conv(4x4, dilation 2) → sigmoid → mask × input
    const m = this.conv(x, s.final, [1, 1, 1, 1], 2);
    const o = new Float32Array(X.length);
    for (let i = 0; i < o.length; i++) o[i] = X[i] / (1 + Math.exp(-m.data[i]));
    return o;
  }
}

function elu(a) { for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < 0) a[i] = Math.exp(v) - 1; } }

function bnInPlace(t, bn, net) {
  const sc = net.f32(bn.scale, bn.n), sh = net.f32(bn.shift, bn.n), C = bn.n, d = t.data;
  for (let i = 0; i < d.length; i += C) for (let c = 0; c < C; c++) d[i + c] = d[i + c] * sc[c] + sh[c];
}
function concatC(a, b) {
  const [, H, W, Ca] = a.shape, Cb = b.shape[3], C = Ca + Cb, n = H * W;
  const o = new Float32Array(n * C);
  for (let p = 0; p < n; p++) { o.set(a.data.subarray(p * Ca, p * Ca + Ca), p * C); o.set(b.data.subarray(p * Cb, p * Cb + Cb), p * C + Ca); }
  return T([1, H, W, C], o);
}

/** Transposed convolution, stride 2, 'SAME' (TF Conv2DBackpropInput). w: [KH, KW, CO, CI]. */
export function convTranspose2d(x, w, bias, s = 2) {
  const [, H, W, CI] = x.shape, [KH, KW, CO, CI2] = w.shape;
  if (CI !== CI2) throw new Error('convT channel mismatch');
  const OH = H * s, OW = W * s;
  const pt = Math.floor(Math.max((H - 1) * s + KH - OH, 0) / 2), pl = Math.floor(Math.max((W - 1) * s + KW - OW, 0) / 2);
  const out = new Float32Array(OH * OW * CO);
  for (let p = 0; p < OH * OW; p++) out.set(bias, p * CO);
  const xd = x.data, wd = w.data;
  for (let iy = 0; iy < H; iy++) for (let ix = 0; ix < W; ix++) {
    const xb = (iy * W + ix) * CI;
    for (let ky = 0; ky < KH; ky++) {
      const oy = iy * s - pt + ky; if (oy < 0 || oy >= OH) continue;
      for (let kx = 0; kx < KW; kx++) {
        const ox = ix * s - pl + kx; if (ox < 0 || ox >= OW) continue;
        const ob = (oy * OW + ox) * CO, wb = (ky * KW + kx) * CO * CI;
        for (let co = 0; co < CO; co++) {
          const q = wb + co * CI;
          let a0 = 0, a1 = 0, a2 = 0, a3 = 0, ci = 0;
          for (; ci + 3 < CI; ci += 4) {
            a0 += xd[xb + ci] * wd[q + ci]; a1 += xd[xb + ci + 1] * wd[q + ci + 1];
            a2 += xd[xb + ci + 2] * wd[q + ci + 2]; a3 += xd[xb + ci + 3] * wd[q + ci + 3];
          }
          for (; ci < CI; ci++) a0 += xd[xb + ci] * wd[q + ci];
          out[ob + co] += a0 + a1 + a2 + a3;
        }
      }
    }
  }
  return T([1, OH, OW, CO], out);
}
