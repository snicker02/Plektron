// Web Worker: harmonic/percussive separation off the main thread.
import { harmonic } from './hpss.js';

self.onmessage = (e) => {
  const { x } = e.data;
  try {
    const y = harmonic(x, {}, (p) => self.postMessage({ type: 'progress', p }));
    self.postMessage({ type: 'done', y }, [y.buffer]);
  } catch (err) { self.postMessage({ type: 'error', message: String(err && err.stack || err) }); }
};
