// WebAssembly SIMD convolution kernels (hand-written WAT in tools/vecmat.wat).
// Falls back to null when the browser lacks WASM SIMD; callers then use the JS kernels.
import { VECMAT_WASM } from './simd-bytes.js';

const b64 = (s) => (typeof atob === 'function' ? Uint8Array.from(atob(s), (c) => c.charCodeAt(0)) : new Uint8Array(Buffer.from(s, 'base64')));

export function createSimd() {
  let inst;
  try {
    const bytes = b64(VECMAT_WASM);
    if (!WebAssembly.validate(bytes)) return null;
    inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
  } catch (e) { return null; }
  const mem = inst.exports.mem, vecmat = inst.exports.vecmat;
  let heap = new Float32Array(mem.buffer);
  const ensure = (floats) => {
    const need = floats * 4 + 64;
    if (mem.buffer.byteLength < need) { mem.grow(Math.ceil((need - mem.buffer.byteLength) / 65536)); }
    heap = new Float32Array(mem.buffer);
  };

  /** conv2d NHWC × HWIO, dilation 1, SAME or VALID; CO % 16 === 0. */
  function conv2d(x, w, strides, padding, bias) {
    const [N, H, W, C] = x.shape, [KH, KW, CI, CO] = w.shape;
    const sh = strides[1], sw = strides[2];
    let OH, OW, pt = 0, pl = 0;
    if (padding === 'SAME') {
      OH = Math.ceil(H / sh); OW = Math.ceil(W / sw);
      pt = Math.floor(Math.max((OH - 1) * sh + KH - H, 0) / 2); pl = Math.floor(Math.max((OW - 1) * sw + KW - W, 0) / 2);
    } else { OH = Math.floor((H - KH) / sh) + 1; OW = Math.floor((W - KW) / sw) + 1; }
    const nx = x.data.length, nw = w.data.length, no = N * OH * OW * CO;
    ensure(nx + nw + no);
    const X0 = 0, W0 = nx, O0 = nx + nw;
    heap.set(x.data, X0); heap.set(w.data, W0);
    for (let p = 0; p < N * OH * OW; p++) { if (bias) heap.set(bias, O0 + p * CO); else heap.fill(0, O0 + p * CO, O0 + p * CO + CO); }
    for (let n = 0; n < N; n++) for (let oh = 0; oh < OH; oh++) for (let ow = 0; ow < OW; ow++) {
      const iw0 = ow * sw - pl, kw0 = Math.max(0, -iw0), kw1 = Math.min(KW, W - iw0);
      if (kw1 <= kw0) continue;
      const oo = O0 + ((n * OH + oh) * OW + ow) * CO;
      for (let kh = 0; kh < KH; kh++) {
        const ih = oh * sh - pt + kh; if (ih < 0 || ih >= H) continue;
        const xb = X0 + ((n * H + ih) * W + iw0 + kw0) * C;
        const wb = W0 + (kh * KW + kw0) * CI * CO;
        vecmat(xb * 4, wb * 4, oo * 4, (kw1 - kw0) * C, CO);
      }
    }
    return { shape: [N, OH, OW, CO], data: heap.slice(O0, O0 + no), dtype: 'float32' };
  }

  /** Transposed conv, stride s, SAME. w laid out [KH, KW, CI, CO] (note: CI before CO). CO % 16 === 0. */
  function convT(x, w, bias, s = 2) {
    const [, H, W, CI] = x.shape, [KH, KW, , CO] = w.shape;
    const OH = H * s, OW = W * s;
    const pt = Math.floor(Math.max((H - 1) * s + KH - OH, 0) / 2), pl = Math.floor(Math.max((W - 1) * s + KW - OW, 0) / 2);
    const nx = x.data.length, nw = w.data.length, no = OH * OW * CO;
    ensure(nx + nw + no);
    const X0 = 0, W0 = nx, O0 = nx + nw;
    heap.set(x.data, X0); heap.set(w.data, W0);
    for (let p = 0; p < OH * OW; p++) heap.set(bias, O0 + p * CO);
    for (let iy = 0; iy < H; iy++) for (let ix = 0; ix < W; ix++) {
      const xb = (X0 + (iy * W + ix) * CI) * 4;
      for (let ky = 0; ky < KH; ky++) {
        const oy = iy * s - pt + ky; if (oy < 0 || oy >= OH) continue;
        for (let kx = 0; kx < KW; kx++) {
          const ox = ix * s - pl + kx; if (ox < 0 || ox >= OW) continue;
          vecmat(xb, (W0 + (ky * KW + kx) * CI * CO) * 4, (O0 + (oy * OW + ox) * CO) * 4, CI, CO);
        }
      }
    }
    return { shape: [1, OH, OW, CO], data: heap.slice(O0, O0 + no), dtype: 'float32' };
  }
  return { conv2d, convT };
}
