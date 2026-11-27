// Piano-roll view: model activation heatmap + detected notes, beat grid, editing.
import { attachGestures, fitCanvas, STRING_COLORS } from './gestures.js';
import { N_PITCH, MIDI_OFFSET, midiName } from '../engine/constants.js';

export class RollView {
  constructor(canvas, app) {
    this.cv = canvas; this.ctx = canvas.getContext('2d'); this.app = app;
    this.t0 = 0; this.pps = 120; this.data = null; this.heat = null; this.playT = -1; this.dirty = true; this.hits = [];
    attachGestures(canvas, {
      drag: (dx) => { this.panBy(dx); },
      pan: (dx) => { this.panBy(dx); },
      zoom: (f, cx) => { const t = this.xToT(cx); this.pps = Math.max(15, Math.min(1200, this.pps * f)); this.t0 = t - (cx - this.gx) / this.pps; this.clamp(); this.dirty = true; this.draw(); },
      wheel: (dx, dy, shift) => { this.panBy(-(shift ? dy : dx || dy) ); },
      tap: (x, y) => this.tap(x, y),
      doubleTap: (x, y) => this.doubleTap(x, y),
    });
    new ResizeObserver(() => { this.dirty = true; this.draw(); }).observe(canvas);
  }
  get gx() { return 34; }
  xToT(x) { return this.t0 + (x - this.gx) / this.pps; }
  tToX(t) { return this.gx + (t - this.t0) * this.pps; }
  panBy(dx) { this.t0 -= dx / this.pps; this.clamp(); this.app.userScrolled && this.app.userScrolled(); this.dirty = true; this.draw(); }
  clamp() { const dur = this.data ? this.data.duration : 0; this.t0 = Math.max(-0.5, Math.min(dur, this.t0)); }
  pitchRange() { const d = this.data; const lo = d.tuning[0] + d.capo - 1, hi = d.tuning[d.tuning.length - 1] + d.capo + d.maxFret + 1; return [lo, Math.min(hi, MIDI_OFFSET + N_PITCH - 1)]; }
  yToPitch(y) { const [lo, hi] = this.pitchRange(); const rh = (this.h - 16) / (hi - lo + 1); return Math.round(hi - (y - 2) / rh + 0.5 - 0.5); }
  pitchY(p) { const [lo, hi] = this.pitchRange(); const rh = (this.h - 16) / (hi - lo + 1); return 2 + (hi - p) * rh; }

  setModel(model) {
    // build a heat image (time × pitch) in chunks to respect canvas size limits
    this.heat = []; if (!model) return;
    const T = model.T, CH = 4096;
    for (let a = 0; a < T; a += CH) {
      const wpx = Math.min(CH, T - a), cv = document.createElement('canvas'); cv.width = wpx; cv.height = N_PITCH;
      const cx = cv.getContext('2d'), img = cx.createImageData(wpx, N_PITCH);
      for (let t = 0; t < wpx; t++) for (let p = 0; p < N_PITCH; p++) {
        const v = model.frames[(a + t) * N_PITCH + p], o = ((N_PITCH - 1 - p) * wpx + t) * 4;
        const g = Math.min(1, v * 1.4);
        img.data[o] = 40 + 150 * g; img.data[o + 1] = 70 + 60 * g; img.data[o + 2] = 120 + 40 * g; img.data[o + 3] = Math.round(255 * Math.pow(g, 0.8));
      }
      cx.putImageData(img, 0, 0);
      this.heat.push({ cv, a, n: wpx, t0: model.frameTimes[a], t1: model.frameTimes[a + wpx - 1] });
    }
    this.dirty = true;
  }
  setData(d) { this.data = d; this.dirty = true; this.draw(); }
  setPlayhead(t) { if (Math.abs(t - this.playT) > 1e-3) { this.playT = t; this.dirty = true; } }
  ensureVisible(t) {
    const x = this.tToX(t);
    if (x < this.gx + 10 || x > this.w - 30) { this.t0 = t - (this.w - this.gx) * 0.2 / this.pps; this.clamp(); this.dirty = true; }
  }

  tap(x, y) {
    for (const h of this.hits) if (x >= h.x - 3 && x <= h.x + h.w + 3 && y >= h.y - 3 && y <= h.y + h.h + 3) { this.app.select(h.note); return; }
    this.app.select(null);
    if (x > this.gx) this.app.seek(Math.max(0, this.xToT(x)));
  }
  doubleTap(x, y) {
    if (x <= this.gx) return;
    for (const h of this.hits) if (x >= h.x - 3 && x <= h.x + h.w + 3 && y >= h.y - 3 && y <= h.y + h.h + 3) return;
    const [lo, hi] = this.pitchRange(), rh = (this.h - 16) / (hi - lo + 1);
    const p = Math.round(hi - (y - 2 - rh / 2) / rh);
    this.app.addNote(p, this.xToT(x));
  }

  draw() {
    if (!this.dirty) return; this.dirty = false;
    const { w, h, dpr } = fitCanvas(this.cv); this.w = w; this.h = h;
    const c = this.ctx; c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#0c0e11'; c.fillRect(0, 0, w, h);
    this.hits = [];
    const d = this.data; if (!d) return;
    const [lo, hi] = this.pitchRange(), rh = (h - 16) / (hi - lo + 1);
    const tA = this.xToT(this.gx), tB = this.xToT(w);
    // pitch lanes
    for (let p = lo; p <= hi; p++) {
      const y = 2 + (hi - p) * rh;
      const black = [1, 3, 6, 8, 10].includes(p % 12);
      c.fillStyle = black ? '#0a0c0e' : '#101317'; c.fillRect(this.gx, y, w - this.gx, rh);
      if (p % 12 === 0) { c.fillStyle = '#20262d'; c.fillRect(this.gx, y + rh - 1, w - this.gx, 1); }
    }
    // heat
    c.imageSmoothingEnabled = true;
    for (const hc of this.heat || []) {
      if (hc.t1 < tA || hc.t0 > tB) continue;
      const pLoIdx = lo - MIDI_OFFSET, pHiIdx = hi - MIDI_OFFSET;
      const sy = N_PITCH - 1 - pHiIdx, sh = pHiIdx - pLoIdx + 1;
      const x0 = this.tToX(hc.t0), x1 = this.tToX(hc.t1 + (hc.t1 - hc.t0) / Math.max(1, hc.n - 1));
      c.globalAlpha = 0.85;
      c.drawImage(hc.cv, 0, Math.max(0, sy), hc.n, Math.min(sh, N_PITCH), x0, 2, x1 - x0, sh * rh);
      c.globalAlpha = 1;
    }
    // beats / bars
    d.beats.forEach((bt, i) => {
      if (bt < tA - 1 || bt > tB + 1) return;
      const x = this.tToX(bt), isBar = d.isBarBeat(i);
      c.fillStyle = isBar ? 'rgba(242,162,58,.35)' : 'rgba(255,255,255,.07)'; c.fillRect(x, 0, 1, h - 14);
      if (isBar) { c.fillStyle = '#8b96a3'; c.font = '10px system-ui, sans-serif'; c.textAlign = 'left'; c.fillText(String(d.barOfBeat(i)), x + 3, h - 4); }
    });
    // notes
    const sel = this.app.selected;
    for (const n of d.notes) {
      if (n.t + n.dur < tA || n.t > tB) continue;
      const x = this.tToX(n.t), ww = Math.max(3, n.dur * this.pps), y = 2 + (hi - n.pitch) * rh;
      const f = d.fingerOf.get(n.key);
      const col = f && f.string >= 0 ? STRING_COLORS[(d.tuning.length - 1 - f.string) % STRING_COLORS.length] : null;
      const isSel = sel && sel.key === n.key;
      c.fillStyle = isSel ? '#ffd166' : (col || 'rgba(160,170,180,.25)');
      c.globalAlpha = col || isSel ? 0.35 + 0.65 * Math.min(1, n.amp * 1.4) : 1;
      c.fillRect(x, y + 1, ww, Math.max(2, rh - 2));
      c.globalAlpha = 1;
      if (!col && !isSel) { c.strokeStyle = '#8b96a3'; c.setLineDash([3, 2]); c.strokeRect(x + 0.5, y + 1.5, ww - 1, Math.max(2, rh - 3)); c.setLineDash([]); }
      if (rh >= 9 && ww > 14 && f && f.string >= 0) { c.fillStyle = '#111'; c.font = `600 ${Math.min(11, rh - 1)}px ui-monospace, monospace`; c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText(String(f.fret), x + 2, y + rh / 2 + 0.5); }
      this.hits.push({ x, y, w: ww, h: rh, note: n });
    }
    // playhead
    if (this.playT >= 0) { const x = this.tToX(this.playT); c.fillStyle = '#f2a23a'; c.fillRect(x - 0.75, 0, 1.5, h - 14); }
    // keyboard gutter
    c.fillStyle = '#161a1f'; c.fillRect(0, 0, this.gx, h);
    c.font = '9px system-ui, sans-serif'; c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let p = lo; p <= hi; p++) {
      const y = 2 + (hi - p) * rh;
      if (p % 12 === 0 || (rh > 10 && [4, 7, 9].includes(p % 12))) { c.fillStyle = p % 12 === 0 ? '#c3cad3' : '#6f7b88'; c.fillText(midiName(p), this.gx - 4, y + rh / 2); }
    }
  }
}
