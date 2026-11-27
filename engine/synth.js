// Karplus–Strong plucked-string rendering of the transcription (for A/B listening).
export function renderPluck(notes, duration, sr = 22050, release = 0.25) {
  const out = new Float32Array(Math.ceil((duration + 2) * sr));
  let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
  for (const n of notes) {
    const f = 440 * Math.pow(2, (n.pitch - 69) / 12);
    const P = Math.max(2, Math.round(sr / f));
    const buf = new Float32Array(P);
    for (let i = 0; i < P; i++) buf[i] = rnd();
    // soften the excitation
    for (let k = 0; k < 2; k++) for (let i = 1; i < P; i++) buf[i] = 0.5 * (buf[i] + buf[i - 1]);
    const start = Math.floor(n.t * sr);
    const len = Math.min(out.length - start, Math.floor((Math.max(0.15, n.dur) + release) * sr));
    const relStart = Math.floor(Math.max(0.15, n.dur) * sr);
    const gain = 0.25 * (0.35 + 0.65 * Math.min(1, n.amp * 1.6));
    const decay = 0.996 - Math.min(0.01, f / 200000);
    let idx = 0;
    for (let i = 0; i < len; i++) {
      const a = buf[idx], b = buf[(idx + 1) % P];
      buf[idx] = decay * 0.5 * (a + b);
      let env = 1; if (i > relStart) env = Math.max(0, 1 - (i - relStart) / (release * sr));
      out[start + i] += a * gain * env;
      idx = (idx + 1) % P;
    }
  }
  // soft clip
  for (let i = 0; i < out.length; i++) { const v = out[i]; out[i] = v / (1 + Math.abs(v)); }
  return out;
}
