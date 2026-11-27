// Plektron — minimal TensorFlow.js graph-model interpreter (no dependencies).
// Implements only the ops used by Spotify's Basic Pitch model (Apache-2.0).
// Tensors: { shape: number[], data: Float32Array | Int32Array, dtype: 'float32' | 'int32' }

const b64 = (s) => (typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary'));

function prod(a) { let p = 1; for (const v of a) p *= v; return p; }
function stridesOf(shape) {
  const s = new Array(shape.length); let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
  return s;
}
function T(shape, data, dtype) { return { shape, data, dtype: dtype || (data instanceof Int32Array ? 'int32' : 'float32') }; }
function alloc(shape, dtype) { const n = prod(shape); return T(shape, dtype === 'int32' ? new Int32Array(n) : new Float32Array(n), dtype || 'float32'); }
function vals(t) { return Array.from(t.data); }
function normAxis(a, rank) { return a < 0 ? a + rank : a; }

function attrI(node, k, d) { const a = node.attr && node.attr[k]; if (!a) return d; if (a.i !== undefined) return parseInt(a.i, 10); return d; }
function attrS(node, k, d) { const a = node.attr && node.attr[k]; if (!a || a.s === undefined) return d; return b64(a.s); }
function attrList(node, k) { const a = node.attr && node.attr[k]; if (!a || !a.list) return null; if (a.list.i) return a.list.i.map(Number); if (a.list.s) return a.list.s.map(b64); return []; }
function attrB(node, k, d) { const a = node.attr && node.attr[k]; if (!a) return d; if (a.b !== undefined) return !!a.b; return d; }

// ---------- generic elementwise / broadcast ----------
function unary(t, f) { const o = alloc(t.shape.slice(), 'float32'); const a = t.data, b = o.data; for (let i = 0; i < a.length; i++) b[i] = f(a[i]); return o; }

function broadcastShape(a, b) {
  const r = Math.max(a.length, b.length), out = new Array(r);
  for (let i = 0; i < r; i++) {
    const x = a[a.length - r + i] ?? 1, y = b[b.length - r + i] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new Error(`broadcast mismatch ${a} vs ${b}`);
    out[i] = Math.max(x, y);
  }
  return out;
}
function binary(A, B, f) {
  const shape = broadcastShape(A.shape, B.shape), r = shape.length;
  const o = alloc(shape, A.dtype === 'int32' && B.dtype === 'int32' ? 'int32' : 'float32');
  const n = o.data.length, ad = A.data, bd = B.data;
  if (ad.length === n && bd.length === n) { for (let i = 0; i < n; i++) o.data[i] = f(ad[i], bd[i]); return o; }
  if (bd.length === 1) { const y = bd[0]; for (let i = 0; i < n; i++) o.data[i] = f(ad[i % ad.length], y); if (ad.length === n) return o; }
  if (ad.length === 1 && bd.length === n) { const x = ad[0]; for (let i = 0; i < n; i++) o.data[i] = f(x, bd[i]); return o; }
  const pa = new Array(r).fill(1), pb = new Array(r).fill(1);
  for (let i = 0; i < A.shape.length; i++) pa[r - A.shape.length + i] = A.shape[i];
  for (let i = 0; i < B.shape.length; i++) pb[r - B.shape.length + i] = B.shape[i];
  const sa = stridesOf(pa), sb = stridesOf(pb);
  for (let i = 0; i < r; i++) { if (pa[i] === 1) sa[i] = 0; if (pb[i] === 1) sb[i] = 0; }
  const idx = new Array(r).fill(0);
  let ia = 0, ib = 0;
  for (let k = 0; k < n; k++) {
    o.data[k] = f(ad[ia], bd[ib]);
    for (let d = r - 1; d >= 0; d--) {
      idx[d]++; ia += sa[d]; ib += sb[d];
      if (idx[d] < shape[d]) break;
      ia -= sa[d] * shape[d]; ib -= sb[d] * shape[d]; idx[d] = 0;
    }
  }
  return o;
}

// ---------- shape ops ----------
function transpose(t, perm) {
  const r = t.shape.length, shape = perm.map((p) => t.shape[p]);
  const o = alloc(shape, t.dtype), is = stridesOf(t.shape);
  const ps = perm.map((p) => is[p]);
  const n = o.data.length, idx = new Array(r).fill(0); let src = 0;
  const id = t.data, od = o.data;
  for (let k = 0; k < n; k++) {
    od[k] = id[src];
    for (let d = r - 1; d >= 0; d--) {
      idx[d]++; src += ps[d];
      if (idx[d] < shape[d]) break;
      src -= ps[d] * shape[d]; idx[d] = 0;
    }
  }
  return o;
}

function pad(t, pads, mode) { // pads: [[before, after], ...]; mode 'CONSTANT' | 'REFLECT' | 'SYMMETRIC'
  const r = t.shape.length, shape = t.shape.map((s, i) => s + pads[i][0] + pads[i][1]);
  const o = alloc(shape, t.dtype), is = stridesOf(t.shape), n = o.data.length;
  const idx = new Array(r).fill(0);
  for (let k = 0; k < n; k++) {
    let src = 0, ok = true;
    for (let d = 0; d < r; d++) {
      let x = idx[d] - pads[d][0]; const L = t.shape[d];
      if (x < 0 || x >= L) {
        if (mode === 'CONSTANT') { ok = false; break; }
        if (mode === 'REFLECT') { while (x < 0 || x >= L) { if (x < 0) x = -x; if (x >= L) x = 2 * (L - 1) - x; } }
        else { while (x < 0 || x >= L) { if (x < 0) x = -x - 1; if (x >= L) x = 2 * L - 1 - x; } }
      }
      src += x * is[d];
    }
    if (ok) o.data[k] = t.data[src];
    for (let d = r - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
  }
  return o;
}

function concat(ts, axis) {
  const r = ts[0].shape.length; axis = normAxis(axis, r);
  const shape = ts[0].shape.slice(); shape[axis] = ts.reduce((s, t) => s + t.shape[axis], 0);
  const o = alloc(shape, ts[0].dtype);
  const outer = prod(shape.slice(0, axis)), inner = prod(shape.slice(axis + 1));
  let off = 0;
  for (const t of ts) {
    const w = t.shape[axis] * inner;
    for (let a = 0; a < outer; a++) o.data.set(t.data.subarray(a * w, a * w + w), a * shape[axis] * inner + off);
    off += w;
  }
  return o;
}

function stack(ts, axis) {
  const r = ts[0].shape.length + 1; axis = normAxis(axis, r);
  const ex = ts.map((t) => { const s = t.shape.slice(); s.splice(axis, 0, 1); return T(s, t.data, t.dtype); });
  return concat(ex, axis);
}

function stridedSlice(t, begin, end, strides, node) {
  const bm = attrI(node, 'begin_mask', 0), em = attrI(node, 'end_mask', 0);
  const nm = attrI(node, 'new_axis_mask', 0), sm = attrI(node, 'shrink_axis_mask', 0);
  if (attrI(node, 'ellipsis_mask', 0)) throw new Error('ellipsis_mask unsupported');
  const dims = []; // {inDim, start, step, count, keep}
  let inD = 0;
  for (let i = 0; i < begin.length; i++) {
    if (nm & (1 << i)) { dims.push({ newAxis: true }); continue; }
    const L = t.shape[inD], s = strides[i];
    let b = begin[i], e = end[i];
    if (sm & (1 << i)) {
      if (b < 0) b += L;
      dims.push({ inDim: inD, start: b, step: 1, count: 1, keep: false }); inD++; continue;
    }
    if (bm & (1 << i)) b = s > 0 ? 0 : L - 1; else { if (b < 0) b += L; b = s > 0 ? Math.min(Math.max(b, 0), L) : Math.min(Math.max(b, -1), L - 1); }
    if (em & (1 << i)) e = s > 0 ? L : -1; else { if (e < 0) e += L; e = s > 0 ? Math.min(Math.max(e, 0), L) : Math.min(Math.max(e, -1), L - 1); }
    const count = Math.max(0, s > 0 ? Math.ceil((e - b) / s) : Math.ceil((b - e) / -s));
    dims.push({ inDim: inD, start: b, step: s, count, keep: true }); inD++;
  }
  for (; inD < t.shape.length; inD++) dims.push({ inDim: inD, start: 0, step: 1, count: t.shape[inD], keep: true });
  const is = stridesOf(t.shape);
  const it = dims.filter((d) => !d.newAxis);
  const iterShape = it.map((d) => d.count);
  const outShape = []; for (const d of dims) { if (d.newAxis) outShape.push(1); else if (d.keep) outShape.push(d.count); }
  const o = alloc(outShape, t.dtype), n = o.data.length;
  const idx = new Array(it.length).fill(0);
  for (let k = 0; k < n; k++) {
    let src = 0;
    for (let j = 0; j < it.length; j++) src += (it[j].start + idx[j] * it[j].step) * is[it[j].inDim];
    o.data[k] = t.data[src];
    for (let d = it.length - 1; d >= 0; d--) { if (++idx[d] < iterShape[d]) break; idx[d] = 0; }
  }
  return o;
}

function reduce(t, axesT, keep, init, f) {
  const r = t.shape.length;
  const axes = new Set(vals(axesT).map((a) => normAxis(a, r)));
  const outFull = t.shape.map((s, i) => (axes.has(i) ? 1 : s));
  const o = alloc(outFull, 'float32'); o.data.fill(init);
  const os = stridesOf(outFull).map((s, i) => (axes.has(i) ? 0 : s));
  const idx = new Array(r).fill(0), n = t.data.length; let dst = 0;
  for (let k = 0; k < n; k++) {
    o.data[dst] = f(o.data[dst], t.data[k]);
    for (let d = r - 1; d >= 0; d--) {
      idx[d]++; dst += os[d];
      if (idx[d] < t.shape[d]) break;
      dst -= os[d] * t.shape[d]; idx[d] = 0;
    }
  }
  if (!keep) o.shape = t.shape.filter((_, i) => !axes.has(i));
  return o;
}

// ---------- convolution (NHWC, filter HWIO) ----------
export function conv2d(x, w, strides, padding, dilations, bias, relu) {
  const [N, H, W, C] = x.shape, [KH, KW, CI, CO] = w.shape;
  if (CI !== C) throw new Error(`conv channel mismatch ${C} vs ${CI}`);
  const sh = strides[1], sw = strides[2], dh = dilations[1], dw = dilations[2];
  const ekh = (KH - 1) * dh + 1, ekw = (KW - 1) * dw + 1;
  let OH, OW, pt = 0, pl = 0;
  if (padding === 'SAME') {
    OH = Math.ceil(H / sh); OW = Math.ceil(W / sw);
    pt = Math.floor(Math.max((OH - 1) * sh + ekh - H, 0) / 2);
    pl = Math.floor(Math.max((OW - 1) * sw + ekw - W, 0) / 2);
  } else { OH = Math.floor((H - ekh) / sh) + 1; OW = Math.floor((W - ekw) / sw) + 1; }
  const o = alloc([N, OH, OW, CO]);
  const xd = x.data, wd = w.data, od = o.data;
  const rowW = CI * CO;
  if (dw === 1) {
    // fast path: for a fixed kh the valid kw range is contiguous in memory (NHWC), as are the weights
    const acc = new Float32Array(CO);
    for (let n = 0; n < N; n++) for (let oh = 0; oh < OH; oh++) for (let ow = 0; ow < OW; ow++) {
      const iw0 = ow * sw - pl;
      const kw0 = Math.max(0, -iw0), kw1 = Math.min(KW, W - iw0);
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
      if (CO !== 8 && CO !== 1) { if (bias) for (let c = 0; c < CO; c++) acc[c] = bias[c]; else acc.fill(0); }
      for (let kh = 0; kh < KH; kh++) {
        const ih = oh * sh - pt + kh * dh; if (ih < 0 || ih >= H || kw1 <= kw0) continue;
        const xb = ((n * H + ih) * W + iw0 + kw0) * C, cnt = (kw1 - kw0) * C;
        const wb = (kh * KW + kw0) * rowW;
        if (CO === 8) {
          for (let j = 0; j < cnt; j++) {
            const xv = xd[xb + j], q = wb + j * 8;
            a0 += xv * wd[q]; a1 += xv * wd[q + 1]; a2 += xv * wd[q + 2]; a3 += xv * wd[q + 3];
            a4 += xv * wd[q + 4]; a5 += xv * wd[q + 5]; a6 += xv * wd[q + 6]; a7 += xv * wd[q + 7];
          }
        } else if (CO === 1) {
          for (let j = 0; j < cnt; j++) a0 += xd[xb + j] * wd[wb + j];
        } else if (CO % 8 === 0) {
          for (let cb = 0; cb < CO; cb += 8) {
            let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, b7 = 0;
            for (let j = 0; j < cnt; j++) {
              const xv = xd[xb + j], q = wb + j * CO + cb;
              b0 += xv * wd[q]; b1 += xv * wd[q + 1]; b2 += xv * wd[q + 2]; b3 += xv * wd[q + 3];
              b4 += xv * wd[q + 4]; b5 += xv * wd[q + 5]; b6 += xv * wd[q + 6]; b7 += xv * wd[q + 7];
            }
            acc[cb] += b0; acc[cb + 1] += b1; acc[cb + 2] += b2; acc[cb + 3] += b3;
            acc[cb + 4] += b4; acc[cb + 5] += b5; acc[cb + 6] += b6; acc[cb + 7] += b7;
          }
        } else {
          for (let j = 0; j < cnt; j++) {
            const xv = xd[xb + j]; if (xv === 0) continue;
            const q = wb + j * CO;
            for (let c = 0; c < CO; c++) acc[c] += xv * wd[q + c];
          }
        }
      }
      const oo = ((n * OH + oh) * OW + ow) * CO;
      if (CO === 8) {
        const b = bias;
        const v = [a0, a1, a2, a3, a4, a5, a6, a7];
        for (let c = 0; c < 8; c++) { const r = v[c] + (b ? b[c] : 0); od[oo + c] = relu && r < 0 ? 0 : r; }
      } else if (CO === 1) { const r = a0 + (bias ? bias[0] : 0); od[oo] = relu && r < 0 ? 0 : r; }
      else for (let c = 0; c < CO; c++) { const r = acc[c]; od[oo + c] = relu && r < 0 ? 0 : r; }
    }
    return o;
  }
  const acc = new Float64Array(CO);
  for (let n = 0; n < N; n++) {
    for (let oh = 0; oh < OH; oh++) {
      for (let ow = 0; ow < OW; ow++) {
        if (bias) for (let c = 0; c < CO; c++) acc[c] = bias[c]; else acc.fill(0);
        for (let kh = 0; kh < KH; kh++) {
          const ih = oh * sh - pt + kh * dh; if (ih < 0 || ih >= H) continue;
          for (let kw = 0; kw < KW; kw++) {
            const iw = ow * sw - pl + kw * dw; if (iw < 0 || iw >= W) continue;
            const xo = ((n * H + ih) * W + iw) * C;
            const wo = (kh * KW + kw) * rowW;
            for (let ci = 0; ci < C; ci++) {
              const xv = xd[xo + ci]; const wr = wo + ci * CO;
              for (let c = 0; c < CO; c++) acc[c] += xv * wd[wr + c];
            }
          }
        }
        const oo = ((n * OH + oh) * OW + ow) * CO;
        for (let c = 0; c < CO; c++) { const v = acc[c]; od[oo + c] = relu && v < 0 ? 0 : v; }
      }
    }
  }
  return o;
}

// ---------- graph ----------
const sig = (v) => 1 / (1 + Math.exp(-v));

export class GraphModel {
  constructor(json, weightBuffer) {
    this.nodes = new Map();
    for (const n of json.modelTopology.node) this.nodes.set(n.name, n);
    this.consts = new Map();
    let off = 0;
    for (const group of json.weightsManifest) {
      for (const w of group.weights) {
        const n = prod(w.shape); const bytes = n * 4;
        const slice = weightBuffer.slice(off, off + bytes); off += bytes;
        const data = w.dtype === 'int32' ? new Int32Array(slice) : new Float32Array(slice);
        this.consts.set(w.name, T(w.shape.slice(), data, w.dtype === 'int32' ? 'int32' : 'float32'));
      }
    }
    const sigOut = json.signature.outputs;
    this.outputNames = {};
    for (const k of Object.keys(sigOut)) this.outputNames[sigOut[k].name.split(':')[0]] = k;
    this.inputName = Object.values(json.signature.inputs)[0].name.split(':')[0];
    this.order = this._topo(Object.keys(this.outputNames));
  }

  _topo(outputs) {
    const seen = new Set(), order = [];
    const visit = (name) => {
      if (seen.has(name)) return; seen.add(name);
      const n = this.nodes.get(name); if (!n) throw new Error('missing node ' + name);
      for (const inp of n.input || []) { if (inp.startsWith('^')) continue; visit(inp.split(':')[0]); }
      order.push(name);
    };
    outputs.forEach(visit);
    // reference counts for freeing intermediates
    this.refs = new Map();
    for (const name of order) for (const inp of this.nodes.get(name).input || []) {
      if (inp.startsWith('^')) continue; const k = inp.split(':')[0]; this.refs.set(k, (this.refs.get(k) || 0) + 1);
    }
    return order;
  }

  // input: Float32Array of the model's input shape (batch 1). Returns map outputName -> tensor
  execute(inputData, inputShape) {
    const env = new Map(); const refs = new Map(this.refs);
    const get = (name) => { const k = name.split(':')[0]; const t = env.get(k); if (!t) throw new Error('unresolved ' + k); return t; };
    const release = (name) => {
      const k = name.split(':')[0]; const c = refs.get(k) - 1; refs.set(k, c);
      if (c <= 0 && !this.outputNames[k]) env.delete(k);
    };
    for (const name of this.order) {
      const node = this.nodes.get(name);
      const ins = (node.input || []).filter((s) => !s.startsWith('^'));
      env.set(name, this._run(node, ins.map(get), inputData, inputShape));
      ins.forEach(release);
    }
    const out = {};
    for (const k of Object.keys(this.outputNames)) out[this.outputNames[k]] = env.get(k);
    return out;
  }

  _run(node, I, inputData, inputShape) {
    const op = node.op;
    switch (op) {
      case 'Const': { const c = this.consts.get(node.name); if (!c) throw new Error('no const ' + node.name); return c; }
      case 'Placeholder': return T(inputShape.slice(), inputData, 'float32');
      case 'Identity': return I[0];
      case 'Shape': return T([I[0].shape.length], Int32Array.from(I[0].shape), 'int32');
      case 'Reshape': {
        const t = I[0]; const s = vals(I[1]); const n = t.data.length;
        const known = s.reduce((p, v) => (v === -1 ? p : p * v), 1);
        return T(s.map((v) => (v === -1 ? n / known : v)), t.data, t.dtype);
      }
      case 'ExpandDims': {
        const t = I[0]; const r = t.shape.length; let a = vals(I[1])[0]; if (a < 0) a += r + 1;
        const s = t.shape.slice(); s.splice(a, 0, 1); return T(s, t.data, t.dtype);
      }
      case 'Squeeze': {
        const t = I[0]; const r = t.shape.length; const dims = attrList(node, 'squeeze_dims');
        const drop = new Set(dims && dims.length ? dims.map((d) => normAxis(d, r)) : t.shape.map((s, i) => (s === 1 ? i : -1)).filter((i) => i >= 0));
        return T(t.shape.filter((_, i) => !drop.has(i)), t.data, t.dtype);
      }
      case 'Transpose': return transpose(I[0], vals(I[1]));
      case 'Pad': { const p = vals(I[1]); const pads = []; for (let i = 0; i < p.length; i += 2) pads.push([p[i], p[i + 1]]); return pad(I[0], pads, 'CONSTANT'); }
      case 'MirrorPad': { const p = vals(I[1]); const pads = []; for (let i = 0; i < p.length; i += 2) pads.push([p[i], p[i + 1]]); return pad(I[0], pads, attrS(node, 'mode', 'REFLECT')); }
      case 'StridedSlice': return stridedSlice(I[0], vals(I[1]), vals(I[2]), vals(I[3]), node);
      case 'Pack': {
        const ax = attrI(node, 'axis', 0);
        if (I[0].shape.length === 0) return T([I.length], I[0].dtype === 'int32' ? Int32Array.from(I.map((t) => t.data[0])) : Float32Array.from(I.map((t) => t.data[0])), I[0].dtype);
        return stack(I, ax);
      }
      case 'ConcatV2': { const ax = vals(I[I.length - 1])[0]; return concat(I.slice(0, -1), ax); }
      case 'Neg': return unary(I[0], (v) => -v);
      case 'Square': return unary(I[0], (v) => v * v);
      case 'Sqrt': return unary(I[0], Math.sqrt);
      case 'Log': return unary(I[0], Math.log);
      case 'Sigmoid': return unary(I[0], sig);
      case 'Mul': return binary(I[0], I[1], (a, b) => a * b);
      case 'Add': case 'AddV2': return binary(I[0], I[1], (a, b) => a + b);
      case 'Sub': return binary(I[0], I[1], (a, b) => a - b);
      case 'DivNoNan': return binary(I[0], I[1], (a, b) => (b === 0 ? 0 : a / b));
      case 'Sum': return reduce(I[0], I[1], attrB(node, 'keep_dims', false), 0, (a, b) => a + b);
      case 'Min': return reduce(I[0], I[1], attrB(node, 'keep_dims', false), Infinity, Math.min);
      case 'Max': return reduce(I[0], I[1], attrB(node, 'keep_dims', false), -Infinity, Math.max);
      case 'Conv2D': case '_FusedConv2D': {
        if (attrS(node, 'data_format', 'NHWC') !== 'NHWC') throw new Error('NCHW unsupported');
        const fused = op === '_FusedConv2D' ? attrList(node, 'fused_ops') : [];
        const hasBias = fused.includes('BiasAdd'), relu = fused.includes('Relu');
        if (fused.some((f) => f !== 'BiasAdd' && f !== 'Relu')) throw new Error('fused op unsupported: ' + fused);
        return conv2d(I[0], I[1], attrList(node, 'strides'), attrS(node, 'padding', 'VALID'), attrList(node, 'dilations') || [1, 1, 1, 1], hasBias ? I[2].data : null, relu);
      }
      default: throw new Error('Unsupported op ' + op);
    }
  }
}
