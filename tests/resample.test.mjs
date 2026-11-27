// resampler: passband tone kept, out-of-band tone suppressed
globalThis.window = {};
const { resample } = await import('../engine/transcribe.js');
const sr = 44100, n = sr;
const mk = (f) => Float32Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * f * i / sr));
const rms = (a) => Math.sqrt(a.slice(1000, -1000).reduce((s, v) => s + v * v, 0) / (a.length - 2000));
const pass = resample(mk(440), sr, 22050), stop = resample(mk(15000), sr, 22050);
console.log('len', pass.length, 'pass rms', rms(pass).toFixed(3), '(0.707 ideal)  15 kHz alias rms', rms(stop).toFixed(4));
let z = 0; for (let i = 1001; i < pass.length - 1000; i++) if (pass[i - 1] < 0 && pass[i] >= 0) z++;
console.log('440 Hz zero crossings / s ≈', (z / ((pass.length - 2000) / 22050)).toFixed(1));
