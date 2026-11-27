// Beat-synchronous chord recognition from the model's note posteriors:
// chroma + bass chroma → template scores → Viterbi smoothing with a change penalty.
import { N_PITCH, MIDI_OFFSET } from './constants.js';

export const ROOTS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
export const QUALITIES = [
  { q: '', iv: [0, 4, 7], prior: 0 },
  { q: 'm', iv: [0, 3, 7], prior: 0 },
  { q: '7', iv: [0, 4, 7, 10], prior: -0.04 },
  { q: 'maj7', iv: [0, 4, 7, 11], prior: -0.05 },
  { q: 'm7', iv: [0, 3, 7, 10], prior: -0.04 },
  { q: 'sus2', iv: [0, 2, 7], prior: -0.06 },
  { q: 'sus4', iv: [0, 5, 7], prior: -0.06 },
  { q: 'dim', iv: [0, 3, 6], prior: -0.08 },
  { q: '5', iv: [0, 7], prior: -0.03 },
];

export function chordLabels() {
  const L = [];
  for (let r = 0; r < 12; r++) for (const Q of QUALITIES) {
    const tpl = new Float32Array(12);
    for (const i of Q.iv) tpl[(r + i) % 12] = i === 0 ? 1.25 : 1;
    let nrm = 0; for (const v of tpl) nrm += v * v; nrm = Math.sqrt(nrm);
    for (let k = 0; k < 12; k++) tpl[k] /= nrm;
    L.push({ name: ROOTS[r] + Q.q, root: r, quality: Q.q, pcs: Q.iv.map((i) => (r + i) % 12), tpl, prior: Q.prior });
  }
  return L;
}

/**
 * @param frames T*88 note posteriors
 * @param frameTimes per-frame time (s)
 * @param beats beat times (s)
 * @param opts {changePenalty, noChordThresh}
 * @returns per-beat labels and merged segments
 */
export function recognizeChords(frames, frameTimes, beats, opts = {}) {
  const T = frameTimes.length, labels = chordLabels(), NL = labels.length;
  const penalty = opts.changePenalty ?? 0.18;
  const ncThresh = opts.noChordThresh ?? 0.15;
  const nB = Math.max(0, beats.length);
  const bounds = beats.slice(); bounds.push(beats.length ? beats[beats.length - 1] + (beats.length > 1 ? beats[beats.length - 1] - beats[beats.length - 2] : 0.5) : 0);
  const emit = []; // per beat: Float32Array(NL+1) (last = N.C.)
  let fi = 0;
  for (let b = 0; b < nB; b++) {
    const t0 = bounds[b], t1 = bounds[b + 1];
    const ch = new Float64Array(12), bass = new Float64Array(12); let energy = 0, cnt = 0;
    while (fi < T && frameTimes[fi] < t0) fi++;
    let f = fi;
    for (; f < T && frameTimes[f] < t1; f++) {
      cnt++;
      let lowest = -1;
      for (let p = 0; p < N_PITCH; p++) {
        const v = frames[f * N_PITCH + p]; if (v < 0.08) continue;
        const pc = (p + MIDI_OFFSET) % 12; ch[pc] += v; energy += v;
        if (lowest < 0 && v > 0.25) lowest = p;
      }
      if (lowest >= 0) bass[(lowest + MIDI_OFFSET) % 12] += 1;
    }
    let nrm = 0; for (const v of ch) nrm += v * v; nrm = Math.sqrt(nrm) || 1;
    let bs = 0; for (const v of bass) bs += v; bs = bs || 1;
    const e = new Float32Array(NL + 1);
    const level = cnt ? energy / cnt : 0;
    for (let l = 0; l < NL; l++) {
      const L = labels[l]; let s = 0;
      for (let k = 0; k < 12; k++) s += (ch[k] / nrm) * L.tpl[k];
      s += 0.25 * bass[L.root] / bs + L.prior;
      e[l] = s;
    }
    e[NL] = level < ncThresh ? 1 : 0.2; // no chord
    emit.push(e);
  }
  // Viterbi
  const S = NL + 1, score = new Float64Array(S), back = [];
  if (nB) for (let s = 0; s < S; s++) score[s] = emit[0][s];
  for (let b = 1; b < nB; b++) {
    let bestPrev = 0; for (let s = 1; s < S; s++) if (score[s] > score[bestPrev]) bestPrev = s;
    const bp = new Int16Array(S), ns = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      const stay = score[s], sw = score[bestPrev] - penalty;
      if (stay >= sw) { ns[s] = stay + emit[b][s]; bp[s] = s; } else { ns[s] = sw + emit[b][s]; bp[s] = bestPrev; }
    }
    score.set(ns); back.push(bp);
  }
  const path = new Array(nB);
  if (nB) {
    let s = 0; for (let k = 1; k < S; k++) if (score[k] > score[s]) s = k;
    for (let b = nB - 1; b >= 0; b--) { path[b] = s; if (b > 0) s = back[b - 1][s]; }
  }
  const perBeat = path.map((s) => (s === NL ? null : labels[s]));
  const segments = [];
  for (let b = 0; b < nB; b++) {
    const L = perBeat[b];
    const last = segments[segments.length - 1];
    if (last && last.label === L) last.endBeat = b + 1;
    else segments.push({ label: L, startBeat: b, endBeat: b + 1 });
  }
  return { perBeat, segments: segments.filter((s) => s.label), labels };
}
