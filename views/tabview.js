// Canvas tab view with virtual scrolling, playhead and note selection.
import { attachGestures, fitCanvas, STRING_COLORS } from './gestures.js';
import { barCount, stringLabel } from '../engine/tab.js';

export class TabView {
  constructor(canvas, app) {
    this.cv = canvas; this.ctx = canvas.getContext('2d'); this.app = app;
    this.scrollY = 0; this.zoom = 1; this.hits = []; this.data = null; this.playSlot = -1; this.dirty = true;
    attachGestures(canvas, {
      drag: (dx, dy) => { this.scrollBy(-dy); },
      pan: (dx, dy) => { this.scrollBy(-dy); },
      zoom: (f) => { this.setZoom(this.zoom * f); app.onZoom && app.onZoom(this.zoom); },
      wheel: (dx, dy) => this.scrollBy(dy),
      tap: (x, y) => this.tap(x, y),
    });
    new ResizeObserver(() => { this.dirty = true; this.draw(); }).observe(canvas);
  }
  setData(d) { this.data = d; this.layout(); this.dirty = true; this.draw(); }
  setZoom(z) { this.zoom = Math.max(0.6, Math.min(2, z)); this.layout(); this.dirty = true; this.draw(); }
  scrollBy(d) { this.scrollY = Math.max(0, Math.min(this.maxScroll(), this.scrollY + d)); this.app.userScrolled && this.app.userScrolled(); this.dirty = true; this.draw(); }
  maxScroll() { return Math.max(0, (this.rows ? this.rows.length * this.rowH + 20 : 0) - (this.h || 0)); }

  layout() {
    const d = this.data; if (!d) return;
    const r = this.cv.getBoundingClientRect(); this.w = r.width; this.h = r.height;
    const S = d.tuning.length, z = this.zoom;
    this.strGap = 15 * z; this.font = Math.round(11.5 * z);
    this.beatW = Math.max(60, 15 * d.q.subdiv) * z; this.cellW = this.beatW / d.q.subdiv;
    this.labelW = 22 * z; this.top = 26 * z;
    this.rowH = this.top + (S - 1) * this.strGap + 20 * z;
    const minBar = this.beatW * d.q.beatsPerBar, avail = this.w - this.labelW - 12;
    this.barsPerRow = Math.max(1, Math.floor(avail / minBar));
    // stretch bars to use the full width (bounded so sparse bars don't look empty)
    this.barW = Math.max(minBar, Math.min(minBar * 1.6, avail / this.barsPerRow));
    this.beatW = this.barW / d.q.beatsPerBar; this.cellW = this.beatW / d.q.subdiv;
    const nb = barCount(d.q);
    this.rows = [];
    for (let b = 0; b < nb; b += this.barsPerRow) this.rows.push(b);
    this.bySlot = new Map(); for (const ev of d.q.events) this.bySlot.set(ev.slot, ev);
    this.chordAt = new Map(); for (const m of d.marks) this.chordAt.set(m.slot, m.name);
    this.scrollY = Math.min(this.scrollY, this.maxScroll());
  }

  slotXY(slot) {
    const spb = this.data.q.slotsPerBar, bar = Math.floor(slot / spb);
    const row = Math.floor(bar / this.barsPerRow), col = bar % this.barsPerRow;
    const x = this.labelW + col * this.barW + (slot - bar * spb + 0.5) * this.cellW;
    return { x, y: row * this.rowH + 10 - this.scrollY, row };
  }

  ensureVisible(slot) {
    if (!this.rows) return;
    const { row } = this.slotXY(slot);
    const y = row * this.rowH + 10;
    if (y < this.scrollY + 4 || y + this.rowH > this.scrollY + this.h - 4) {
      this.scrollY = Math.max(0, Math.min(this.maxScroll(), y - Math.min(this.rowH * 0.5, this.h * 0.2)));
      this.dirty = true;
    }
  }

  setPlayhead(slot) { if (Math.abs(slot - this.playSlot) > 0.01) { this.playSlot = slot; this.dirty = true; } }

  tap(x, y) {
    for (const h of this.hits) if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) { this.app.select(h.note); return; }
    // otherwise seek to the slot under the finger
    if (!this.data) return;
    const row = Math.floor((y + this.scrollY - 10) / this.rowH); if (row < 0 || row >= this.rows.length) return;
    const col = Math.floor((x - this.labelW) / this.barW); if (col < 0 || col >= this.barsPerRow) return;
    const bar = this.rows[row] + col; const within = Math.floor((x - this.labelW - col * this.barW) / this.cellW);
    this.app.select(null);
    this.app.seekSlot(bar * this.data.q.slotsPerBar + within);
  }

  draw() {
    if (!this.dirty) return; this.dirty = false;
    const { w, h, dpr } = fitCanvas(this.cv);
    if (Math.abs(w - (this.w || 0)) > 0.5 || Math.abs(h - (this.h || 0)) > 0.5) this.layout();
    const c = this.ctx; c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#0c0e11'; c.fillRect(0, 0, w, h);
    this.hits = [];
    const d = this.data; if (!d) return;
    const S = d.tuning.length, spb = d.q.slotsPerBar, nb = barCount(d.q), sel = this.app.selected;
    const r0 = Math.max(0, Math.floor((this.scrollY - 10) / this.rowH)), r1 = Math.min(this.rows.length - 1, Math.ceil((this.scrollY + h) / this.rowH));
    c.textBaseline = 'middle';
    for (let r = r0; r <= r1; r++) {
      const y0 = r * this.rowH + 10 - this.scrollY, ys = y0 + this.top;
      const b0 = this.rows[r], b1 = Math.min(nb, b0 + this.barsPerRow);
      const xEnd = this.labelW + (b1 - b0) * this.barW;
      // playhead column
      if (this.playSlot >= 0) {
        const pb = Math.floor(this.playSlot / spb);
        if (pb >= b0 && pb < b1) {
          const px = this.labelW + (pb - b0) * this.barW + (this.playSlot - pb * spb) * this.cellW;
          c.fillStyle = 'rgba(242,162,58,.16)'; c.fillRect(px - this.cellW * 0.5, ys - 10, this.cellW, (S - 1) * this.strGap + 20);
          c.fillStyle = '#f2a23a'; c.fillRect(px - 0.75, ys - 12, 1.5, (S - 1) * this.strGap + 24);
        }
      }
      // strings
      c.lineWidth = 1; c.strokeStyle = '#3a434d';
      for (let s = 0; s < S; s++) { const yy = ys + (S - 1 - s) * this.strGap; c.beginPath(); c.moveTo(this.labelW, yy + 0.5); c.lineTo(xEnd, yy + 0.5); c.stroke(); }
      c.fillStyle = '#6f7b88'; c.font = `${Math.round(10 * this.zoom)}px ui-monospace, Menlo, monospace`; c.textAlign = 'center';
      for (let s = 0; s < S; s++) c.fillText(stringLabel(d.tuning[s]), this.labelW * 0.45, ys + (S - 1 - s) * this.strGap);
      for (let b = b0; b <= b1; b++) {
        const x = this.labelW + (b - b0) * this.barW;
        c.strokeStyle = '#6f7b88'; c.beginPath(); c.moveTo(x + 0.5, ys); c.lineTo(x + 0.5, ys + (S - 1) * this.strGap); c.stroke();
        if (b < b1) {
          c.fillStyle = '#56616d'; c.textAlign = 'left'; c.font = `${Math.round(9 * this.zoom)}px system-ui, sans-serif`;
          c.fillText(String(b + 1), x + 2, y0 + 6);
          // beat ticks
          c.strokeStyle = '#262d35';
          for (let k = 1; k < d.q.beatsPerBar; k++) { const bx = x + k * this.beatW; c.beginPath(); c.moveTo(bx + 0.5, ys + (S - 1) * this.strGap + 5); c.lineTo(bx + 0.5, ys + (S - 1) * this.strGap + 9); c.stroke(); }
        }
      }
      // chords + notes
      c.textAlign = 'center';
      for (let b = b0; b < b1; b++) for (let k = 0; k < spb; k++) {
        const slot = b * spb + k, x = this.labelW + (b - b0) * this.barW + (k + 0.5) * this.cellW;
        const ch = this.chordAt.get(slot);
        if (ch) { c.fillStyle = '#f2a23a'; c.font = `600 ${Math.round(11 * this.zoom)}px system-ui, sans-serif`; c.textAlign = 'left'; c.fillText(ch, x - this.cellW * 0.4, y0 + 15 * this.zoom); c.textAlign = 'center'; }
        const ev = this.bySlot.get(slot); if (!ev) continue;
        c.font = `600 ${this.font}px ui-monospace, Menlo, monospace`;
        for (const n of ev.notes) {
          if (n.string < 0) continue;
          const yy = ys + (S - 1 - n.string) * this.strGap, txt = String(n.fret);
          const tw = c.measureText(txt).width + 4, th = this.strGap - 2;
          const isSel = sel && sel.key === n.key;
          c.fillStyle = isSel ? '#ffd166' : '#0c0e11';
          c.fillRect(x - tw / 2, yy - th / 2, tw, th);
          c.fillStyle = isSel ? '#111' : (n.locked ? '#4fc3c9' : (n.added ? '#9fe29d' : '#e6e9ee'));
          c.fillText(txt, x, yy + 0.5);
          // sustain line
          if ((n.len || 1) > 1) { c.strokeStyle = 'rgba(230,233,238,.18)'; c.beginPath(); const x2 = x + Math.min(n.len - 1, spb - k - 0.5) * this.cellW; c.moveTo(x + tw / 2, yy + th / 2 - 1); c.lineTo(x2, yy + th / 2 - 1); c.stroke(); }
          this.hits.push({ x: x - Math.max(tw, 14) / 2, y: yy - th / 2 - 2, w: Math.max(tw, 14), h: th + 4, note: n });
        }
      }
    }
  }
}
