// Note-event creation from Basic Pitch posteriors.
// Port of basic-pitch's outputToNotesPoly (Apache-2.0, Spotify AB) to flat typed arrays,
// with the "melodia trick" rewritten to run in O(n log n) instead of O(n * T * 88).
import { N_PITCH, MIDI_OFFSET } from './constants.js';

/**
 * @param {Float32Array} framesIn  T*88 note posteriors
 * @param {Float32Array} onsetsIn  T*88 onset posteriors
 * @param {number} T  number of frames
 * @param {object} o  options
 * @returns {{start:number,end:number,pitch:number,amp:number}[]} frame-indexed notes
 */
export function outputToNotes(framesIn, onsetsIn, T, o = {}) {
  const onsetThresh = o.onsetThresh ?? 0.5;
  const frameThresh = o.frameThresh ?? 0.3;
  const minLen = o.minNoteLen ?? 11;
  const inferOnsets = o.inferOnsets ?? true;
  const melodia = o.melodia ?? true;
  const tol = o.energyTolerance ?? 11;
  const lo = Math.max(0, (o.minMidi ?? MIDI_OFFSET) - MIDI_OFFSET);
  const hi = Math.min(N_PITCH - 1, (o.maxMidi ?? 108) - MIDI_OFFSET);
  const P = N_PITCH;

  const frames = new Float32Array(framesIn.length);
  const onsets = new Float32Array(onsetsIn.length);
  for (let t = 0; t < T; t++) for (let p = lo; p <= hi; p++) {
    frames[t * P + p] = framesIn[t * P + p];
    onsets[t * P + p] = onsetsIn[t * P + p];
  }

  // --- inferred onsets (frame-difference based) ---
  let inferred = onsets;
  if (inferOnsets) {
    const diff = new Float32Array(T * P);
    let dMax = 0, oMax = 0;
    for (let i = 0; i < onsets.length; i++) if (onsets[i] > oMax) oMax = onsets[i];
    for (let t = 2; t < T; t++) for (let p = 0; p < P; p++) {
      const v = frames[t * P + p];
      const d1 = v - frames[(t - 1) * P + p];
      const d2 = v - frames[(t - 2) * P + p];
      const d = Math.max(Math.min(d1, d2), 0);
      diff[t * P + p] = d; if (d > dMax) dMax = d;
    }
    inferred = new Float32Array(T * P);
    const s = dMax > 0 ? oMax / dMax : 0;
    for (let i = 0; i < inferred.length; i++) inferred[i] = Math.max(onsets[i], diff[i] * s);
  }

  // --- peak picking along time ---
  const starts = []; // [t, p]
  for (let t = 0; t < T; t++) for (let p = 0; p < P; p++) {
    const v = inferred[t * P + p];
    if (v <= onsetThresh) continue;
    if (t > 0 && !(v > inferred[(t - 1) * P + p])) continue;
    if (t < T - 1 && !(v > inferred[(t + 1) * P + p])) continue;
    starts.push(t, p);
  }

  const energy = frames.slice();
  const notes = [];
  const clear3 = (j, p) => { energy[j * P + p] = 0; if (p < P - 1) energy[j * P + p + 1] = 0; if (p > 0) energy[j * P + p - 1] = 0; };
  const meanAmp = (a, b, p) => { let s = 0; for (let j = a; j < b; j++) s += frames[j * P + p]; return s / Math.max(1, b - a); };

  // process onsets latest-first (as in the reference implementation)
  for (let k = starts.length - 2; k >= 0; k -= 2) {
    const t0 = starts[k], p = starts[k + 1];
    if (t0 >= T - 1) continue;
    let i = t0 + 1, c = 0;
    while (i < T - 1 && c < tol) { if (energy[i * P + p] < frameThresh) c++; else c = 0; i++; }
    i -= c;
    if (i - t0 <= minLen) continue;
    for (let j = t0; j < i; j++) clear3(j, p);
    let os = 0; for (let j = Math.max(0, t0 - 1); j <= Math.min(T - 1, t0 + 1); j++) os = Math.max(os, onsets[j * P + p]);
    notes.push({ start: t0, end: i, pitch: p + MIDI_OFFSET, amp: meanAmp(t0, i, p), onset: true, os });
  }

  // --- melodia trick: grow notes from remaining energy peaks, strongest first ---
  if (melodia) {
    const cand = [];
    for (let i = 0; i < energy.length; i++) if (energy[i] > frameThresh) cand.push(i);
    cand.sort((a, b) => energy[b] - energy[a]);
    const orig = energy.slice();
    for (const idx of cand) {
      if (energy[idx] !== orig[idx] || energy[idx] <= frameThresh) continue; // already consumed
      const iMid = (idx / P) | 0, p = idx % P;
      energy[idx] = 0;
      let i = iMid + 1, c = 0;
      while (i < T - 1 && c < tol) { if (energy[i * P + p] < frameThresh) c++; else c = 0; clear3(i, p); i++; }
      const iEnd = i - 1 - c;
      i = iMid - 1; c = 0;
      while (i > 0 && c < tol) { if (energy[i * P + p] < frameThresh) c++; else c = 0; clear3(i, p); i--; }
      const iStart = i + 1 + c;
      if (iEnd - iStart <= minLen) continue;
      notes.push({ start: iStart, end: iEnd, pitch: p + MIDI_OFFSET, amp: meanAmp(iStart, iEnd, p), onset: false, os: 0 });
    }
  }
  notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  return notes;
}

/** Convert frame-indexed notes to seconds using the exact per-frame time table. */
export function notesToTime(notes, frameTimes) {
  const last = frameTimes.length - 1;
  const ft = (f) => (f <= last ? frameTimes[f] : frameTimes[last] + (f - last) * (frameTimes[1] - frameTimes[0]));
  return notes.map((n, i) => ({ id: i, t: ft(n.start), dur: ft(n.end) - ft(n.start), pitch: n.pitch, amp: n.amp, onset: n.onset, os: n.os }));
}

/**
 * Cleanup:
 *  - fold onset-less ("melodia") fragments into an adjacent note of the same pitch (sustain re-detections)
 *  - suppress overtones (octave / twelfth / two octaves) of a note in sparse textures
 *  - drop quiet notes, cap polyphony at each onset
 */
export function cleanNotes(notes, { minAmp = 0, maxPoly = 6, overtones = true } = {}) {
  let out = notes.slice().sort((a, b) => a.t - b.t);
  // 1. merge sustain fragments
  const byPitch = new Map();
  for (const n of out) { if (!byPitch.has(n.pitch)) byPitch.set(n.pitch, []); byPitch.get(n.pitch).push(n); }
  const dead = new Set();
  for (const list of byPitch.values()) {
    for (let i = 0; i < list.length; i++) {
      const n = list[i]; if (n.onset !== false || dead.has(n)) continue;
      const prev = list.slice(0, i).reverse().find((x) => !dead.has(x));
      const next = list.slice(i + 1).find((x) => !dead.has(x));
      if (prev && n.t - (prev.t + prev.dur) < 0.09) { prev.dur = Math.max(prev.dur, n.t + n.dur - prev.t); dead.add(n); }
      else if (next && next.onset !== false && next.t - (n.t + n.dur) < 0.06 && n.dur < 0.25) dead.add(n);
    }
  }
  out = out.filter((n) => !dead.has(n));
  // 1b. sustain re-triggers: a same-pitch note restarting right where the previous one ends, exactly when a
  //     *different*, fresh note is attacked, is the old note still ringing (common with let-ring playing).
  const prevEnd = new Map(), fresh = new Map();
  for (const n of out) {
    const pe = prevEnd.get(n.pitch);
    fresh.set(n, !(pe && n.t - pe.end < 0.04));
    if (!pe || n.t + n.dur > pe.end) prevEnd.set(n.pitch, { end: n.t + n.dur, note: n });
  }
  const lastOf = new Map(), merged = new Set();
  for (const n of out) {
    const a = lastOf.get(n.pitch);
    if (a && n.os < 0.68 && n.t - (a.t + a.dur) < 0.04 && n.amp <= a.amp * 1.05) {
      const trigger = out.some((c) => c !== n && c.pitch !== n.pitch && Math.abs(c.t - n.t) < 0.035 && fresh.get(c));
      if (trigger) { a.dur = Math.max(a.dur, n.t + n.dur - a.t); merged.add(n); continue; }
    }
    lastOf.set(n.pitch, n);
  }
  out = out.filter((n) => !merged.has(n));
  // 2. overtone suppression where the onset group is sparse (<= 2 notes)
  if (overtones) {
    const kill = new Set();
    for (const h of out) {
      const group = out.filter((x) => Math.abs(x.t - h.t) < 0.05);
      if (group.length > 2) continue; // chords legitimately contain octaves
      const ratio = 1.15;
      for (const f of out) {
        if (f === h) continue;
        const iv = h.pitch - f.pitch; if (iv !== 12 && iv !== 19 && iv !== 24) continue;
        const ov = Math.min(h.t + h.dur, f.t + f.dur) - Math.max(h.t, f.t);
        if (ov > 0.5 * h.dur && h.t >= f.t - 0.05 && h.amp < f.amp * ratio) { kill.add(h); break; }
      }
    }
    out = out.filter((n) => !kill.has(n));
  }
  out = out.filter((n) => n.amp >= minAmp);
  if (maxPoly > 0) {
    const keep = new Array(out.length).fill(true);
    for (let i = 0; i < out.length; i++) {
      const live = [];
      for (let j = 0; j < out.length && out[j].t <= out[i].t + 1e-6; j++) if (keep[j] && out[j].t + out[j].dur > out[i].t) live.push(j);
      if (live.length > maxPoly) {
        live.sort((a, b) => out[a].amp - out[b].amp);
        for (let k = 0; k < live.length - maxPoly; k++) keep[live[k]] = false;
      }
    }
    out = out.filter((_, i) => keep[i]);
  }
  out.forEach((n, i) => { n.id = i; });
  return out;
}
