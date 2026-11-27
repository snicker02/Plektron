// AI source separation driver: STFT (2048 / hop 512 at 22.05 kHz — the same time/frequency grid
// Spleeter uses at 44.1 kHz), U-Net masks from a worker pool, masked inverse STFT.
// Only bins below ~5.5 kHz go through the network (all the note detector needs); the mask of the
// top analysed bin is extended above that.
import { fft } from './beats.js';

const N = 2048, HOP = 512, SEG = 512, BINS = 512, MAG_SCALE = 2; // ×2 matches 4096-pt @ 44.1 kHz magnitudes
export const SEP_MODEL_MB = 43;

function hann() { const w = new Float64Array(N); for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N); return w; }

function stftFrame(x, m, w, re, im) {
  const c = m * HOP - N / 2;
  for (let i = 0; i < N; i++) { const k = c + i; re[i] = k >= 0 && k < x.length ? x[k] * w[i] : 0; im[i] = 0; }
  fft(re, im);
}

let assets = null;
/** Fetch the model weights (split into parts < 20 MB for GitHub uploads), cached via the Cache API. */
export async function loadSeparationModel(base, onProgress) {
  if (assets) return assets;
  const manifest = await (await fetch(base + 'spleeter4.json')).json();
  const parts = manifest.parts || [{ file: 'spleeter4.bin', bytes: manifest.bytes }];
  let cache = null;
  try { cache = await caches.open('plektron-models-v1'); } catch (e) { cache = null; }
  const buf = new Uint8Array(manifest.bytes);
  let got = 0;
  for (const part of parts) {
    const url = new URL(base + part.file, location.href).href;
    let resp = cache ? await cache.match(url) : null;
    if (!resp) {
      resp = await fetch(url);
      if (!resp.ok) throw new Error(`could not download ${part.file} (${resp.status})`);
      if (cache) { try { await cache.put(url, resp.clone()); } catch (e) {} }
    }
    if (resp.body && resp.body.getReader) {
      const reader = resp.body.getReader();
      for (;;) { const { done, value } = await reader.read(); if (done) break; buf.set(value, got); got += value.length; onProgress && onProgress(got / manifest.bytes); }
    } else { const b = new Uint8Array(await resp.arrayBuffer()); buf.set(b, got); got += b.length; onProgress && onProgress(got / manifest.bytes); }
  }
  if (got !== manifest.bytes) throw new Error(`separation model incomplete (${got} of ${manifest.bytes} bytes) — check that all ${parts.length} spleeter4-*.bin files were uploaded`);
  assets = { manifest, bin: buf.buffer };
  return assets;
}

function poolSize() {
  const cores = navigator.hardwareConcurrency || 4;
  const lowMem = (navigator.deviceMemory && navigator.deviceMemory <= 4) || matchMedia('(pointer: coarse)').matches;
  return Math.max(1, Math.min(cores - 1, lowMem ? 2 : 4));
}

/**
 * Separate the "other" stem (no vocals, drums or bass).
 * @param {Float32Array[]} ch 1 or 2 channels at 22.05 kHz
 * @param {(phase:string, p:number)=>void} onProgress
 * @returns {Promise<Float32Array[]>} stereo (or mono) stem at 22.05 kHz
 */
export async function separateOther(ch, onProgress, token = {}, base = './model/', keep = 'other') {
  const { manifest, bin } = await loadSeparationModel(base, (p) => onProgress && onProgress('download', p));
  const L = ch[0], R = ch[1] || ch[0], len = L.length;
  const F = Math.ceil(len / HOP) + 1, nSeg = Math.ceil(F / SEG);
  const w = hann(), re = new Float64Array(N), im = new Float64Array(N);
  // magnitude segments [SEG][BINS][2]
  const segX = [];
  for (let s = 0; s < nSeg; s++) {
    const X = new Float32Array(SEG * BINS * 2);
    for (let t = 0; t < SEG; t++) {
      const m = s * SEG + t; if (m >= F) break;
      for (let c = 0; c < 2; c++) {
        stftFrame(c ? R : L, m, w, re, im);
        for (let k = 0; k < BINS; k++) X[(t * BINS + k) * 2 + c] = MAG_SCALE * Math.hypot(re[k], im[k]);
      }
    }
    segX.push(X);
  }
  // masks from the worker pool
  const masks = new Array(nSeg);
  const nW = Math.min(poolSize(), nSeg);
  const workers = [];
  let next = 0, done = 0;
  try {
    await new Promise((resolve, reject) => {
      const feed = (wk) => {
        if (token.cancelled) { reject(new Error('cancelled')); return; }
        if (next >= nSeg) return;
        const id = next++, X = segX[id];
        wk.postMessage({ type: 'run', id, X, H: SEG, W: BINS, keep });
      };
      for (let i = 0; i < nW; i++) {
        const wk = new Worker(new URL('./separate-worker.js', import.meta.url), { type: 'module' });
        workers.push(wk);
        wk.onerror = (e) => reject(new Error('separation worker failed: ' + (e.message || 'module workers unsupported?')));
        wk.onmessage = (e) => {
          const m = e.data;
          if (m.type === 'ready') feed(wk);
          else if (m.type === 'done') {
            masks[m.id] = m.mask; done++; onProgress && onProgress('separate', done / nSeg);
            if (done === nSeg) resolve(); else feed(wk);
          } else reject(new Error(m.message));
        };
        wk.postMessage({ type: 'init', manifest, bin: bin.slice(0) });
      }
    });
  } finally { workers.forEach((x) => x.terminate()); }
  // masked inverse STFT with overlap-add
  const outs = [], C = ch.length > 1 ? 2 : 1;
  for (let c = 0; c < C; c++) {
    const x = c ? R : L, y = new Float64Array(len), norm = new Float64Array(len);
    for (let m = 0; m < F; m++) {
      const s = (m / SEG) | 0, t = m % SEG, mk = masks[s];
      stftFrame(x, m, w, re, im);
      let top = 0; for (let k = BINS - 16; k < BINS; k++) top += mk[(t * BINS + k) * 2 + c]; top /= 16;
      for (let k = 0; k <= N / 2; k++) {
        const g = k < BINS ? mk[(t * BINS + k) * 2 + c] : top;
        re[k] *= g; im[k] *= g;
        if (k > 0 && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; }
      }
      for (let i = 0; i < N; i++) im[i] = -im[i];
      fft(re, im);
      const c0 = m * HOP - N / 2;
      for (let i = 0; i < N; i++) { const k = c0 + i; if (k < 0 || k >= len) continue; y[k] += (re[i] / N) * w[i]; norm[k] += w[i] * w[i]; }
      if ((m & 255) === 0) { onProgress && onProgress('rebuild', (c * F + m) / (C * F)); await new Promise((r) => setTimeout(r, 0)); if (token.cancelled) throw new Error('cancelled'); }
    }
    const o = new Float32Array(len);
    for (let i = 0; i < len; i++) o[i] = norm[i] > 1e-6 ? y[i] / norm[i] : 0;
    outs.push(o);
  }
  return outs;
}
