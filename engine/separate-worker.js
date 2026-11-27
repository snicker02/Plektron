// Web Worker: runs the Spleeter 4-stem U-Net on one magnitude segment and returns the soft
// mask of the "other" stem (everything that isn't vocals, drums or bass — guitars, keys…).
import { UNet4 } from './unet.js';

let net = null;
self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') { net = new UNet4(m.manifest, m.bin); self.postMessage({ type: 'ready', simd: !!net.simd }); return; }
    if (m.type === 'run') {
      const { X, H, W } = m;
      const est = net.run(X, H, W);
      const keep = m.keep || 'other';
      const mask = new Float32Array(X.length);
      const names = net.stems;
      for (let i = 0; i < X.length; i++) {
        let sum = 1e-10;
        for (const n of names) { const v = est[n][i]; sum += v * v; }
        let num = 0; for (const k of keep.split('+')) { const v = est[k][i]; num += v * v + 2.5e-11; }
        mask[i] = num / sum;
      }
      self.postMessage({ type: 'done', id: m.id, mask }, [mask.buffer]);
    }
  } catch (err) { self.postMessage({ type: 'error', id: m.id, message: String(err && err.stack || err) }); }
};
