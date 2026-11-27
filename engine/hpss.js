// Harmonic/percussive separation (Fitzgerald 2010) with soft Wiener masks.
// STFT 1024 / hop 256, sliding medians across time (harmonic) and frequency (percussive).
import { fft } from './beats.js';

const N = 1024, HOP = 256, K = N / 2 + 1;

function hann() { const w = new Float64Array(N); for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N); return w; }

function frameSpectrum(x, m, w, re, im) {
  const c = m * HOP - N / 2;
  for (let i = 0; i < N; i++) { const k = c + i; re[i] = k >= 0 && k < x.length ? x[k] * w[i] : 0; im[i] = 0; }
  fft(re, im);
}

/** Sliding median of window `win` (odd) over a strided sequence, clamped at the edges. */
function slidingMedian(src, off, stride, n, win, dst) {
  const h = win >> 1, sorted = new Float32Array(win);
  const at = (i) => src[off + Math.min(n - 1, Math.max(0, i)) * stride];
  for (let j = 0; j < win; j++) sorted[j] = at(j - h);
  sorted.sort();
  for (let i = 0; i < n; i++) {
    dst[off + i * stride] = sorted[h];
    if (i === n - 1) break;
    const out = at(i - h), inn = at(i + h + 1);
    if (out === inn) continue;
    // remove `out`
    let lo = 0, hi = win - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < out) lo = mid + 1; else hi = mid; }
    // insert `inn` keeping order, shifting across the removed slot
    let p = lo;
    if (inn > out) { while (p < win - 1 && sorted[p + 1] < inn) { sorted[p] = sorted[p + 1]; p++; } }
    else { while (p > 0 && sorted[p - 1] > inn) { sorted[p] = sorted[p - 1]; p--; } }
    sorted[p] = inn;
  }
}

/**
 * Return the harmonic part of x (same length). onProgress(0..1) optional.
 * @param {Float32Array} x
 * @param {{kTime?:number, kFreq?:number, power?:number}} o
 */
export function harmonic(x, o = {}, onProgress) {
  const kt = o.kTime ?? 17, kf = o.kFreq ?? 17, pw = o.power ?? 2;
  const F = Math.ceil(x.length / HOP) + 1;
  const w = hann(), re = new Float64Array(N), im = new Float64Array(N);
  const M = new Float32Array(F * K);
  for (let m = 0; m < F; m++) {
    frameSpectrum(x, m, w, re, im);
    for (let k = 0; k < K; k++) M[m * K + k] = Math.hypot(re[k], im[k]);
    if (onProgress && (m & 1023) === 0) onProgress(0.4 * m / F);
  }
  const H = new Float32Array(F * K), P = new Float32Array(F * K);
  for (let k = 0; k < K; k++) slidingMedian(M, k, K, F, kt, H);
  if (onProgress) onProgress(0.55);
  for (let m = 0; m < F; m++) slidingMedian(M, m * K, 1, K, kf, P);
  if (onProgress) onProgress(0.7);
  const y = new Float64Array(x.length), norm = new Float64Array(x.length);
  for (let m = 0; m < F; m++) {
    frameSpectrum(x, m, w, re, im);
    for (let k = 0; k < K; k++) {
      const h = Math.pow(H[m * K + k], pw), p = Math.pow(P[m * K + k], pw);
      const g = h + p > 0 ? h / (h + p) : 0;
      re[k] *= g; im[k] *= g;
      if (k > 0 && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; }
    }
    // inverse via conjugate trick: x = Re(fft(conj(X))) / N
    for (let i = 0; i < N; i++) im[i] = -im[i];
    fft(re, im);
    const c = m * HOP - N / 2;
    for (let i = 0; i < N; i++) {
      const t = c + i; if (t < 0 || t >= x.length) continue;
      y[t] += (re[i] / N) * w[i]; norm[t] += w[i] * w[i];
    }
    if (onProgress && (m & 1023) === 0) onProgress(0.7 + 0.3 * m / F);
  }
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = norm[i] > 1e-6 ? y[i] / norm[i] : 0;
  return out;
}
