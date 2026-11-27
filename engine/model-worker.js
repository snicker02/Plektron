// Web Worker: runs Basic Pitch windows with the dependency-free graph interpreter.
import { GraphModel } from './graph.js';
import { WIN, HALF_OVERLAP, KEEP_FRAMES, N_PITCH } from './constants.js';

let model = null;

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      model = new GraphModel(m.json, m.weights);
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'run') {
      const out = model.execute(m.data, [1, WIN, 1]);
      const keep = (t, bins) => t.data.slice(HALF_OVERLAP * bins, (HALF_OVERLAP + KEEP_FRAMES) * bins);
      const frames = keep(out['Identity_1:0'], N_PITCH);
      const onsets = keep(out['Identity_2:0'], N_PITCH);
      self.postMessage({ type: 'done', id: m.id, frames, onsets }, [frames.buffer, onsets.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.stack || err) });
  }
};
