// Onset envelope, tempo estimation and dynamic-programming beat tracking (Ellis 2007),
// plus downbeat phase choice and a time→beat-position map used for quantization.
import { SR, FFT_HOP, N_PITCH } from './constants.js';

// ---------- radix-2 FFT (in place) ----------
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

/** Log-spectral-flux onset strength at hop 256 (same frame rate as the model). */
export function spectralFlux(audio) {
  const N = 1024, H = FFT_HOP, nb = 372; // bins up to ~8 kHz
  const nF = Math.max(1, Math.floor(audio.length / H));
  const win = new Float32Array(N); for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  const re = new Float64Array(N), im = new Float64Array(N);
  let prev = new Float32Array(nb), cur = new Float32Array(nb);
  const env = new Float32Array(nF);
  for (let f = 0; f < nF; f++) {
    const c = f * H - N / 2;
    for (let i = 0; i < N; i++) { const k = c + i; re[i] = k >= 0 && k < audio.length ? audio[k] * win[i] : 0; im[i] = 0; }
    fft(re, im);
    let s = 0;
    for (let b = 1; b < nb; b++) {
      const m = Math.log1p(100 * Math.hypot(re[b], im[b]));
      cur[b] = m; const d = m - prev[b]; if (d > 0 && f > 0) s += d;
    }
    env[f] = s; const t = prev; prev = cur; cur = t;
  }
  return env;
}

function normalize(env) {
  // subtract a 0.5 s moving mean, half-wave rectify, scale to unit std
  const n = env.length, w = 22, out = new Float32Array(n);
  let acc = 0; const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { acc += env[i]; pre[i + 1] = acc; }
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - w), b = Math.min(n, i + w + 1);
    out[i] = Math.max(0, env[i] - (pre[b] - pre[a]) / (b - a));
  }
  let m = 0, s = 0; for (const v of out) m += v; m /= n || 1;
  for (const v of out) s += (v - m) * (v - m); s = Math.sqrt(s / (n || 1)) || 1;
  for (let i = 0; i < n; i++) out[i] /= s;
  return out;
}

/** Combined onset envelope: audio flux + pitched onsets from the model. */
export function onsetEnvelope(audio, onsets, T) {
  const flux = normalize(spectralFlux(audio));
  const pitched = new Float32Array(T);
  for (let t = 0; t < T; t++) { let s = 0; for (let p = 0; p < N_PITCH; p++) s += onsets[t * N_PITCH + p]; pitched[t] = s; }
  const pn = normalize(pitched);
  const n = Math.min(flux.length, T), env = new Float32Array(n);
  for (let i = 0; i < n; i++) env[i] = flux[i] + 0.6 * pn[i];
  return env;
}

/** Tempo in BPM via prior-weighted autocorrelation. */
export function estimateTempo(env, fps, { minBpm = 50, maxBpm = 220, center = 110 } = {}) {
  const n = env.length;
  const lagMin = Math.floor(60 * fps / maxBpm), lagMax = Math.ceil(60 * fps / minBpm);
  let best = 120, bestScore = -Infinity;
  const ac = new Float64Array(lagMax + 2);
  for (let lag = lagMin; lag <= lagMax + 1; lag++) { let s = 0; for (let i = lag; i < n; i++) s += env[i] * env[i - lag]; ac[lag] = s / (n - lag); }
  for (let lag = lagMin + 1; lag <= lagMax; lag++) {
    const bpm = 60 * fps / lag;
    const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / center) / 0.9, 2));
    // reinforce with the double-period lag (bar-level periodicity)
    const l2 = Math.min(lagMax + 1, lag * 2);
    const score = (ac[lag] + 0.5 * (ac[l2] || 0)) * prior;
    if (score > bestScore) { bestScore = score; best = lag; }
  }
  // parabolic refinement
  const a = ac[best - 1], b = ac[best], c = ac[best + 1];
  const den = a - 2 * b + c; const off = den !== 0 ? 0.5 * (a - c) / den : 0;
  return 60 * fps / (best + Math.max(-0.5, Math.min(0.5, off)));
}

/** Dynamic-programming beat tracker. Returns beat times in seconds. */
export function trackBeats(env, fps, bpm, tightness = 100) {
  const n = env.length, period = 60 * fps / bpm;
  const score = new Float64Array(n), back = new Int32Array(n).fill(-1);
  const lo = Math.round(period / 2), hi = Math.round(period * 2);
  for (let t = 0; t < n; t++) {
    let best = 0, bi = -1;
    for (let tau = t - hi; tau <= t - lo; tau++) {
      if (tau < 0) continue;
      const d = Math.log((t - tau) / period);
      const v = score[tau] - tightness * d * d;
      if (bi < 0 || v > best) { best = v; bi = tau; }
    }
    score[t] = env[t] + (bi >= 0 ? Math.max(0, best) : 0);
    back[t] = bi >= 0 && best > 0 ? bi : -1;
  }
  // pick end within the last period with the highest score relative to local max
  let end = n - 1, bs = -Infinity;
  for (let t = Math.max(0, n - Math.round(period)); t < n; t++) if (score[t] > bs) { bs = score[t]; end = t; }
  const beats = [];
  for (let t = end; t >= 0; t = back[t]) { beats.push(t); if (back[t] < 0) break; }
  beats.reverse();
  // extend backward to the start with the tempo period
  const out = beats.map((f) => f / fps);
  const ps = period / fps;
  while (out.length && out[0] - ps > -ps * 0.25) out.unshift(Math.max(0, out[0] - ps));
  return out;
}

/** Evenly spaced grid when the user locks a tempo. */
export function fixedBeats(duration, bpm, offset) {
  const p = 60 / bpm, out = [];
  let t = offset % p; if (t < 0) t += p;
  for (; t < duration + p; t += p) out.push(t);
  return out;
}

/**
 * Choose which beat in each bar is the downbeat: favours chord changes, low-register onsets
 * and envelope peaks. `changes` is a per-beat 0/1 array of chord changes.
 */
export function chooseDownbeat(beats, beatsPerBar, notes, env, fps, changes) {
  if (beatsPerBar <= 1 || !beats.length) return 0;
  const pitches = notes.map((n) => n.pitch).sort((a, b) => a - b);
  const lowCut = pitches.length ? pitches[Math.floor(pitches.length * 0.15)] + 4 : 0;
  const score = new Float64Array(beatsPerBar);
  let envMean = 0; for (const v of env) envMean += v; envMean /= env.length || 1;
  let j = 0;
  for (let i = 0; i < beats.length; i++) {
    const f = Math.round(beats[i] * fps);
    let s = f >= 0 && f < env.length ? 0.3 * env[f] / (envMean || 1) : 0;
    while (j < notes.length && notes[j].t < beats[i] - 0.06) j++;
    for (let k = j; k < notes.length && notes[k].t < beats[i] + 0.06; k++) if (notes[k].pitch <= lowCut) s += notes[k].amp;
    if (changes && changes[i]) s += 2;
    score[i % beatsPerBar] += s;
  }
  let best = 0; for (let k = 1; k < beatsPerBar; k++) if (score[k] > score[best]) best = k;
  return best;
}

/** Piecewise-linear time → fractional beat index map (extrapolates at ends). */
export function makeBeatMap(beats) {
  const B = beats.length;
  const p0 = B > 1 ? beats[1] - beats[0] : 0.5, p1 = B > 1 ? beats[B - 1] - beats[B - 2] : 0.5;
  const toPos = (t) => {
    if (B === 0) return t / 0.5;
    if (t <= beats[0]) return (t - beats[0]) / p0;
    if (t >= beats[B - 1]) return B - 1 + (t - beats[B - 1]) / p1;
    let a = 0, b = B - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (beats[m] <= t) a = m; else b = m; }
    return a + (t - beats[a]) / (beats[b] - beats[a]);
  };
  const toTime = (pos) => {
    if (B === 0) return pos * 0.5;
    if (pos <= 0) return beats[0] + pos * p0;
    if (pos >= B - 1) return beats[B - 1] + (pos - (B - 1)) * p1;
    const a = Math.floor(pos); return beats[a] + (pos - a) * (beats[a + 1] - beats[a]);
  };
  return { toPos, toTime };
}

export const FRAME_FPS = SR / FFT_HOP;
