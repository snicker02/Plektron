// drum removal: steady tones survive, clicks vanish
import { harmonic } from '../engine/hpss.js';
const tone = Float32Array.from({ length: 22050 * 3 }, (_, i) => Math.sin(2 * Math.PI * 220 * i / 22050));
const y = harmonic(tone); let e = 0; for (let i = 2000; i < tone.length - 2000; i++) e = Math.max(e, Math.abs(y[i] - tone[i]));
const clicks = new Float32Array(22050 * 3); for (let i = 1000; i < clicks.length; i += 5512) clicks[i] = 1;
const yc = harmonic(clicks); let ec = 0, ei = 0; for (let i = 0; i < clicks.length; i++) { ec += yc[i] ** 2; ei += clicks[i] ** 2; }
console.log(`tone max err ${e.toFixed(4)} (want < 0.01) · click energy kept ${(ec / ei).toFixed(4)} (want < 0.01)`);
if (e > 0.01 || ec / ei > 0.01) process.exit(1);
