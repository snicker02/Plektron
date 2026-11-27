// Tab layout model + ASCII renderer + chord sheet text.
import { NOTE_NAMES } from './constants.js';

export function stringLabel(midi) { return NOTE_NAMES[midi % 12]; }

/** Total bars covering the events. */
export function barCount(q) {
  if (!q.events.length) return 1;
  const last = q.events[q.events.length - 1];
  const end = last.slot + Math.max(1, ...last.notes.map((n) => n.len || 1));
  return Math.max(1, Math.ceil(end / q.slotsPerBar));
}

/** Map chord segments (in beats) to bar-relative slots. */
export function chordMarks(segments, q) {
  // segments use absolute beat indices; slot = beat*subdiv - origin
  return segments.map((s) => ({ slot: s.startBeat * q.subdiv - q.origin, name: s.label.name, label: s.label }))
    .filter((m) => m.slot >= 0);
}

/**
 * Render ASCII tab.
 * @param q quantized structure with fingered events
 * @param tuning low→high midi
 * @param opts {width, chords:[{slot,name}], title, header}
 */
export function renderAscii(q, tuning, opts = {}) {
  const S = tuning.length, width = opts.width ?? 78;
  const nBars = barCount(q), spb = q.slotsPerBar;
  const bySlot = new Map(); for (const ev of q.events) bySlot.set(ev.slot, ev);
  const chordAt = new Map(); for (const c of opts.chords || []) chordAt.set(c.slot, c.name);
  const labels = tuning.map((m) => stringLabel(m));
  const lw = Math.max(...labels.map((l) => l.length));
  // build each bar as column strings
  const bars = [];
  for (let b = 0; b < nBars; b++) {
    const lines = Array.from({ length: S }, () => '');
    let chordLine = '';
    for (let k = 0; k < spb; k++) {
      const slot = b * spb + k, ev = bySlot.get(slot);
      const cell = new Array(S).fill('');
      if (ev) for (const n of ev.notes) if (n.string >= 0) cell[n.string] = String(n.fret);
      const cw = Math.max(1, ...cell.map((c) => c.length));
      const ch = chordAt.get(slot);
      if (ch) {
        const at = lines[0].length + 1; // column of this slot's fret digits
        chordLine = chordLine.length < at ? chordLine.padEnd(at) : chordLine + ' ';
        chordLine += ch;
      }
      for (let s = 0; s < S; s++) lines[s] += '-' + (cell[s] || '').padEnd(cw, '-');
    }
    for (let s = 0; s < S; s++) lines[s] += '-';
    bars.push({ lines, chordLine, w: lines[0].length });
  }
  const out = [];
  if (opts.header) out.push(...opts.header, '');
  let i = 0;
  while (i < bars.length) {
    let w = lw + 1, j = i;
    while (j < bars.length && (j === i || w + bars[j].w + 1 <= width)) { w += bars[j].w + 1; j++; }
    const row = Array.from({ length: S }, (_, s) => labels[s].padEnd(lw) + '|');
    let chordRow = ' '.repeat(lw + 1), any = false;
    for (let k = i; k < j; k++) {
      for (let s = 0; s < S; s++) row[s] += bars[k].lines[s] + '|';
      const cl = bars[k].chordLine; if (cl.trim()) any = true;
      chordRow += cl.padEnd(bars[k].w + 1).slice(0, bars[k].w + 1);
    }
    out.push(`[bar ${i + 1}]`);
    if (any) out.push(chordRow.replace(/\s+$/, ''));
    for (let s = S - 1; s >= 0; s--) out.push(row[s]);
    out.push('');
    i = j;
  }
  return out.join('\n');
}

/** Chord-sheet text: one cell per beat, bars separated by '|'. */
export function renderChordSheet(perBeat, beatsPerBar, downbeat, barsPerLine = 4) {
  const cells = [];
  const pre = (beatsPerBar - (((downbeat % beatsPerBar) + beatsPerBar) % beatsPerBar)) % beatsPerBar;
  for (let i = 0; i < pre; i++) cells.push(null);
  let prev;
  perBeat.forEach((L) => { const name = L ? L.name : 'N.C.'; cells.push(name === prev ? '.' : name); prev = name; });
  const lines = []; let line = '|';
  const nb = Math.ceil(cells.length / beatsPerBar);
  for (let b = 0; b < nb; b++) {
    const bar = cells.slice(b * beatsPerBar, b * beatsPerBar + beatsPerBar).map((c) => (c == null ? ' ' : c).padEnd(5)).join('');
    line += ' ' + bar + '|';
    if ((b + 1) % barsPerLine === 0) { lines.push(line); line = '|'; }
  }
  if (line !== '|') lines.push(line);
  return lines.join('\n');
}
