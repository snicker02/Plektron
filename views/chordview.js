// Chord sheet view: chord diagrams for every chord used + beat grid by bar.
import { chordVoicings } from '../engine/fretboard.js';

export function diagramSVG(v, S, name) {
  const W = 64, H = 78, x0 = 9, y0 = 16, gw = (W - 2 * x0) / Math.max(1, S - 1), fh = 11, nf = 5;
  const fretted = v.frets.filter((f) => f > 0);
  const base = fretted.length && Math.max(...fretted) > 4 ? Math.min(...fretted) : 1;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-label="${name}">`;
  s += `<rect x="${x0}" y="${y0}" width="${gw * (S - 1)}" height="${fh * nf}" fill="none" stroke="#56616d"/>`;
  for (let i = 1; i < S - 1; i++) s += `<line x1="${x0 + i * gw}" y1="${y0}" x2="${x0 + i * gw}" y2="${y0 + fh * nf}" stroke="#56616d"/>`;
  for (let f = 1; f < nf; f++) s += `<line x1="${x0}" y1="${y0 + f * fh}" x2="${x0 + gw * (S - 1)}" y2="${y0 + f * fh}" stroke="#3a434d"/>`;
  if (base === 1) s += `<rect x="${x0 - 1}" y="${y0 - 3}" width="${gw * (S - 1) + 2}" height="3" fill="#c3cad3"/>`;
  else s += `<text x="${W - 1}" y="${y0 + fh * 0.75}" font-size="9" fill="#8b96a3" text-anchor="end">${base}</text>`;
  v.frets.forEach((f, i) => {
    const x = x0 + i * gw;
    if (f < 0) s += `<text x="${x}" y="${y0 - 6}" font-size="9" fill="#8b96a3" text-anchor="middle">×</text>`;
    else if (f === 0) s += `<circle cx="${x}" cy="${y0 - 8}" r="2.6" fill="none" stroke="#c3cad3"/>`;
    else { const row = f - base; s += `<circle cx="${x}" cy="${y0 + (row + 0.5) * fh}" r="3.6" fill="#f2a23a"/>`; }
  });
  return s + '</svg>';
}

export class ChordView {
  constructor(el, app) { this.el = el; this.app = app; this.cells = []; this.curBeat = -1; this.alt = new Map(); }
  setData(d) { this.data = d; this.render(); }
  render() {
    const d = this.data, el = this.el; if (!d) { el.innerHTML = ''; return; }
    const S = d.tuning.length;
    // unique chords in order of appearance
    const seen = new Map();
    for (const seg of d.segments) if (seg.label && !seen.has(seg.label.name)) seen.set(seg.label.name, seg.label);
    let html = '<div class="diagrams">';
    if (S >= 4 && S <= 7) for (const [name, L] of seen) {
      const vs = chordVoicings(L.pcs, L.root, d.tuning, d.capo, 3);
      if (!vs.length) continue;
      const k = (this.alt.get(name) || 0) % vs.length;
      html += `<div class="diag" data-chord="${name}" title="Tap for another voicing"><div class="nm">${name}</div>${diagramSVG(vs[k], S, name)}<div class="alt">${vs.length > 1 ? `${k + 1}/${vs.length}` : '&nbsp;'}</div></div>`;
    }
    html += '</div>';
    if (d.capo) html += `<p class="muted">Capo ${d.capo} — chord shapes are relative to the capo.</p>`;
    html += '<div class="sheet">';
    const bpb = d.beatsPerBar, pre = (bpb - (((d.downbeat % bpb) + bpb) % bpb)) % bpb;
    const cells = []; for (let i = 0; i < pre; i++) cells.push(-1);
    for (let b = 0; b < d.perBeat.length; b++) cells.push(b);
    const nb = Math.ceil(cells.length / bpb);
    let prev = undefined;
    for (let bar = 0; bar < nb; bar++) {
      html += `<div class="cbar" data-bar="${bar}"><div class="n">${bar + 1}</div><div class="beats">`;
      for (let k = 0; k < bpb; k++) {
        const bi = cells[bar * bpb + k];
        if (bi == null || bi < 0) { html += '<div class="bt cont">&nbsp;</div>'; continue; }
        const L = d.perBeat[bi], nm = L ? L.name : 'N.C.';
        const cont = nm === prev; prev = nm;
        html += `<div class="bt${cont ? ' cont' : ''}" data-beat="${bi}">${cont ? '·' : nm}</div>`;
      }
      html += '</div></div>';
    }
    html += '</div>';
    el.innerHTML = html;
    this.cells = [...el.querySelectorAll('.bt[data-beat]')];
    this.bars = [...el.querySelectorAll('.cbar')];
    el.onclick = (e) => {
      const bt = e.target.closest('.bt[data-beat]');
      if (bt) { this.app.seek(d.beats[+bt.dataset.beat]); return; }
      const dg = e.target.closest('.diag');
      if (dg) { const n = dg.dataset.chord; this.alt.set(n, (this.alt.get(n) || 0) + 1); this.render(); }
    };
    this.curBeat = -1;
  }
  setTime(t, follow) {
    const d = this.data; if (!d || !d.beats.length) return;
    let b = -1; for (let i = 0; i < d.beats.length; i++) { if (d.beats[i] <= t + 0.01) b = i; else break; }
    if (b === this.curBeat) return;
    if (this.curBeat >= 0 && this.cells[this.curBeat]) this.cells[this.curBeat].classList.remove('on');
    this.curBeat = b;
    const cell = this.cells[b];
    if (cell) {
      cell.classList.add('on');
      if (follow) {
        const r = cell.getBoundingClientRect(), pr = this.el.getBoundingClientRect();
        if (r.top < pr.top || r.bottom > pr.bottom) this.el.scrollTop += r.top - pr.top - pr.height * 0.3;
      }
    }
  }
}
