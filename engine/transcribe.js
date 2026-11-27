// Audio decode/resample, source isolation (mid/side/L/R + drum removal),
// and windowed model inference across a pool of workers.
import { SR, WIN, HOP, FFT_HOP, OVERLAP_SAMPLES, KEEP_FRAMES, N_PITCH } from './constants.js';
import { fft } from './beats.js';

/** Decode any browser-supported file; return the original buffer and 22.05 kHz channels (1 or 2). */
export async function decodeChannels(arrayBuffer) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  const buf = await new Promise((res, rej) => {
    const p = ctx.decodeAudioData(arrayBuffer.slice(0), res, rej);
    if (p && p.then) p.then(res, rej);
  });
  ctx.close && ctx.close();
  return { original: buf, channels: await toModelRate(buf) };
}

export async function toModelRate(buf) {
  const C = Math.min(2, buf.numberOfChannels);
  try {
    const off = new OfflineAudioContext(C, Math.ceil(buf.duration * SR), SR);
    const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
    const r = await off.startRendering();
    const out = []; for (let c = 0; c < C; c++) out.push(r.getChannelData(c).slice());
    return out;
  } catch (e) {
    // some browsers refuse a 22.05 kHz offline context: resample ourselves
    const out = []; for (let c = 0; c < C; c++) out.push(resample(buf.getChannelData(c), buf.sampleRate, SR));
    return out;
  }
}

/** Windowed-sinc resampler (anti-aliased). */
export function resample(x, from, to) {
  if (from === to) return Float32Array.from(x);
  const ratio = to / from, n = Math.floor(x.length * ratio), out = new Float32Array(n);
  const fc = Math.min(1, ratio) * 0.95, half = 16;
  for (let i = 0; i < n; i++) {
    const c = i / ratio, k0 = Math.floor(c);
    let acc = 0, wsum = 0;
    for (let k = k0 - half + 1; k <= k0 + half; k++) {
      if (k < 0 || k >= x.length) continue;
      const d = c - k, a = Math.PI * d * fc;
      const sinc = d === 0 ? 1 : Math.sin(a) / a;
      const w = 0.5 + 0.5 * Math.cos(Math.PI * d / half);
      const v = sinc * w; acc += x[k] * v; wsum += v;
    }
    out[i] = wsum ? acc / wsum : 0;
  }
  return out;
}

// ---------- source isolation ----------
export const SOURCES = {
  auto: 'Auto',
  ai: 'Guitar & keys (AI separation)',
  mix: 'Full mix',
  side: 'Sides (L−R)',
  left: 'Left channel',
  right: 'Right channel',
};

export function mixOf(ch) {
  if (ch.length === 1) return ch[0];
  const [L, R] = ch, m = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) m[i] = 0.5 * (L[i] + R[i]);
  return m;
}

function sideOf(ch) {
  const [L, R] = ch, s = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) s[i] = 0.5 * (L[i] - R[i]);
  return s;
}

/** Share of spectral energy between 40 and 150 Hz (bass / kick), from up to 60 s of audio. */
function lowShare(x) {
  const N = 4096, hop = 4096, lo = Math.round(40 * N / SR), mid = Math.round(150 * N / SR);
  const end = Math.min(x.length - N, 60 * SR * 2);
  const re = new Float64Array(N), im = new Float64Array(N);
  let a = 0, b = 0;
  for (let s = Math.max(0, Math.floor(x.length / 2 - end / 2)), n = 0; s < x.length - N && n < 400; s += hop * 3, n++) {
    for (let i = 0; i < N; i++) { re[i] = x[s + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)); im[i] = 0; }
    fft(re, im);
    for (let k = lo; k < N / 2; k++) { const p = re[k] * re[k] + im[k] * im[k]; b += p; if (k < mid) a += p; }
  }
  return b > 0 ? a / b : 0;
}

/**
 * Decide what to listen to. Band mixes (a strong, centred bass/kick) go through AI separation;
 * solo or sparse recordings are analysed from the full mix untouched.
 */
export function chooseAuto(ch) {
  if (ch.length < 2) {
    const lm = lowShare(ch[0]);
    return lm > 0.35 ? { mode: 'ai', why: 'heavy low end — looks like a band mix' } : { mode: 'mix', why: 'mono file' };
  }
  const [L, R] = ch;
  let lr = 0, ll = 0, rr = 0;
  for (let i = 0; i < L.length; i += 4) { lr += L[i] * R[i]; ll += L[i] * L[i]; rr += R[i] * R[i]; }
  const corr = lr / Math.sqrt(ll * rr || 1);
  const m = mixOf(ch), s = sideOf(ch);
  let em = 0, es = 0; for (let i = 0; i < m.length; i += 4) { em += m[i] * m[i]; es += s[i] * s[i]; }
  const width = Math.sqrt(es / (em || 1));
  const lm = lowShare(m);
  if (corr > 0.97 || width < 0.08) return lm > 0.35 ? { mode: 'ai', why: 'heavy low end — looks like a band mix' } : { mode: 'mix', why: 'narrow stereo' };
  const ls = lowShare(s);
  if (lm > 0.2 && ls < lm * 0.35) return { mode: 'ai', why: 'band mix — bass and kick are centred' };
  return { mode: 'mix', why: 'stereo recording without a centred rhythm section' };
}

export function sourceSignal(ch, mode) {
  if (mode === 'side' && ch.length > 1) return sideOf(ch);
  if (mode === 'left') return ch[0];
  if (mode === 'right') return ch[ch.length - 1];
  return mixOf(ch);
}

/** Harmonic part of x in a worker (drum removal). */
export function removeDrums(x, onProgress, token = {}) {
  return new Promise((resolve, reject) => {
    const wk = new Worker(new URL('./prep-worker.js', import.meta.url), { type: 'module' });
    const timer = setInterval(() => { if (token.cancelled) { clearInterval(timer); wk.terminate(); reject(new Error('cancelled')); } }, 100);
    const done = (fn) => { clearInterval(timer); wk.terminate(); fn(); };
    wk.onerror = (e) => done(() => reject(new Error('drum removal failed: ' + (e.message || 'worker error'))));
    wk.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') onProgress && onProgress(m.p);
      else if (m.type === 'done') done(() => resolve(m.y));
      else done(() => reject(new Error(m.message)));
    };
    const copy = x.slice();
    wk.postMessage({ x: copy }, [copy.buffer]);
  });
}

// ---------- windowing / inference ----------
/** Split audio into model windows (leading half-overlap of zeros, as Basic Pitch does). */
export function makeWindows(audio) {
  const lead = OVERLAP_SAMPLES / 2;
  const n = Math.max(1, Math.ceil(audio.length / HOP));
  const wins = [];
  for (let w = 0; w < n; w++) {
    const buf = new Float32Array(WIN);
    const start = w * HOP - lead;
    for (let i = 0; i < WIN; i++) { const k = start + i; if (k >= 0 && k < audio.length) buf[i] = audio[k]; }
    wins.push(buf);
  }
  return wins;
}

/** Stitch per-window kept frames into contiguous arrays with an exact time table. */
export function stitch(results, audioLength) {
  const dur = audioLength / SR;
  const times = [];
  for (let w = 0; w < results.length; w++) for (let j = 0; j < KEEP_FRAMES; j++) {
    const t = (w * HOP + j * FFT_HOP) / SR; if (t >= dur) break; times.push(t);
  }
  const T = times.length;
  const frames = new Float32Array(T * N_PITCH), onsets = new Float32Array(T * N_PITCH);
  let f = 0;
  for (let w = 0; w < results.length && f < T; w++) {
    const r = results[w], take = Math.min(KEEP_FRAMES, T - f);
    frames.set(r.frames.subarray(0, take * N_PITCH), f * N_PITCH);
    onsets.set(r.onsets.subarray(0, take * N_PITCH), f * N_PITCH);
    f += take;
  }
  return { frames, onsets, T, frameTimes: Float64Array.from(times) };
}

let modelAssets = null;
async function loadAssets(base) {
  if (modelAssets) return modelAssets;
  const json = await (await fetch(base + 'model.json')).json();
  const weights = await (await fetch(base + json.weightsManifest[0].paths[0])).arrayBuffer();
  modelAssets = { json, weights };
  return modelAssets;
}

/**
 * Run the model over the audio with a worker pool.
 * @param {Float32Array} audio 22050 Hz mono
 * @param {(p:number)=>void} onProgress
 * @param {{cancelled:boolean}} token
 */
export async function runModel(audio, onProgress, token = {}, modelBase = './model/') {
  const { json, weights } = await loadAssets(modelBase);
  const wins = makeWindows(audio);
  const nW = Math.max(1, Math.min(wins.length, Math.max(1, (navigator.hardwareConcurrency || 4) - 1), 8));
  const results = new Array(wins.length);
  let next = 0, done = 0;
  const workers = [];
  try {
    await new Promise((resolve, reject) => {
      const feed = (wk) => {
        if (token.cancelled) { reject(new Error('cancelled')); return; }
        if (next >= wins.length) { if (done === wins.length) resolve(); return; }
        const id = next++;
        wk.postMessage({ type: 'run', id, data: wins[id] }, [wins[id].buffer]);
      };
      for (let i = 0; i < nW; i++) {
        const wk = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
        workers.push(wk);
        wk.onerror = (e) => reject(new Error('worker failed: ' + (e.message || 'module workers unsupported?')));
        wk.onmessage = (e) => {
          const m = e.data;
          if (m.type === 'ready') feed(wk);
          else if (m.type === 'done') {
            results[m.id] = m; done++; onProgress && onProgress(done / wins.length);
            if (done === wins.length) resolve(); else feed(wk);
          } else if (m.type === 'error') reject(new Error(m.message));
        };
        wk.postMessage({ type: 'init', json, weights: weights.slice(0) });
      }
    });
  } finally { workers.forEach((w) => w.terminate()); }
  return stitch(results, audio.length);
}
