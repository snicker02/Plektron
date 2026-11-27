// MIDI (type 0) and MusicXML (tablature staff) writers.
import { barCount } from './tab.js';

// ---------- MIDI ----------
function vlq(n) { const b = [n & 0x7f]; while ((n >>= 7)) b.unshift((n & 0x7f) | 0x80); return b; }
const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const u16 = (n) => [(n >>> 8) & 255, n & 255];

/**
 * @param q quantized structure
 * @param bpm tempo for the tempo meta event
 * @param program GM program (24 = nylon, 25 = steel, 26 = jazz, 27 = clean, 29 = overdrive, 30 = distortion, 33 = bass)
 */
export function toMidi(q, bpm, program = 25) {
  const PPQ = 480, tpSlot = PPQ / q.subdiv;
  const ev = []; // [tick, order, bytes]
  for (const e of q.events) for (const n of e.notes) {
    const on = e.slot * tpSlot, off = (e.slot + (n.len || 1)) * tpSlot - 1;
    const vel = Math.max(20, Math.min(127, Math.round(40 + n.amp * 110)));
    ev.push([on, 1, [0x90, n.pitch, vel]], [off, 0, [0x80, n.pitch, 0]]);
  }
  ev.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const tr = [];
  const mpq = Math.round(60000000 / bpm);
  tr.push(0, 0xff, 0x51, 3, (mpq >> 16) & 255, (mpq >> 8) & 255, mpq & 255);
  tr.push(0, 0xff, 0x58, 4, q.beatsPerBar, 2, 24, 8);
  tr.push(0, 0xc0, program);
  let last = 0;
  for (const [t, , b] of ev) { tr.push(...vlq(Math.max(0, t - last)), ...b); last = Math.max(last, t); }
  tr.push(0, 0xff, 0x2f, 0);
  const bytes = [0x4d, 0x54, 0x68, 0x64, ...u32(6), ...u16(0), ...u16(1), ...u16(PPQ), 0x4d, 0x54, 0x72, 0x6b, ...u32(tr.length), ...tr];
  return new Uint8Array(bytes);
}

// ---------- MusicXML ----------
const STEPS = [['C', 0], ['C', 1], ['D', 0], ['D', 1], ['E', 0], ['F', 0], ['F', 1], ['G', 0], ['G', 1], ['A', 0], ['A', 1], ['B', 0]];
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Split a duration (in slots) into notatable pieces. Returns [{dur, type, dots, tuplet}]
function pieces(len, subdiv) {
  const out = [];
  const beatTypes = [[4, 'whole', 0], [3, 'half', 1], [2, 'half', 0], [1.5, 'quarter', 1], [1, 'quarter', 0], [0.75, 'eighth', 1], [0.5, 'eighth', 0], [0.25, '16th', 0], [0.125, '32nd', 0]];
  if (subdiv === 3 || subdiv === 6) {
    let beats = Math.floor(len / subdiv), r = len - beats * subdiv;
    for (const [b, t, d] of beatTypes) { if (b < 1) break; while (beats >= b) { out.push({ dur: b * subdiv, type: t, dots: d }); beats -= b; } }
    const trip = subdiv === 3 ? [[2, 'quarter'], [1, 'eighth']] : [[4, 'quarter'], [2, 'eighth'], [1, '16th']];
    for (const [u, t] of trip) while (r >= u) { out.push({ dur: u, type: t, dots: 0, tuplet: true }); r -= u; }
    return out;
  }
  let r = len;
  for (const [b, t, d] of beatTypes) { const u = b * subdiv; if (u < 1 || !Number.isInteger(u)) continue; while (r >= u) { out.push({ dur: u, type: t, dots: d }); r -= u; } }
  return out;
}

export function toMusicXML(q, tuning, { capo = 0, title = 'Plektron transcription', bpm = 120 } = {}) {
  const S = tuning.length, nBars = barCount(q), spb = q.slotsPerBar, div = q.subdiv;
  const bySlot = new Map(); for (const e of q.events) bySlot.set(e.slot, e);
  const L = [];
  L.push('<?xml version="1.0" encoding="UTF-8" standalone="no"?>');
  L.push('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">');
  L.push('<score-partwise version="3.1">');
  L.push(`<work><work-title>${esc(title)}</work-title></work>`);
  L.push('<identification><encoding><software>Plektron</software></encoding></identification>');
  L.push('<part-list><score-part id="P1"><part-name>Guitar</part-name></score-part></part-list>');
  L.push('<part id="P1">');
  const noteXml = (pitch, pc, chord, string, fret, tieStart, tieStop) => {
    const [step, alter] = STEPS[pitch % 12], oct = Math.floor(pitch / 12) - 1;
    let x = '<note>' + (chord ? '<chord/>' : '') + `<pitch><step>${step}</step>${alter ? '<alter>1</alter>' : ''}<octave>${oct}</octave></pitch>`;
    x += `<duration>${pc.dur}</duration>`;
    if (tieStop) x += '<tie type="stop"/>';
    if (tieStart) x += '<tie type="start"/>';
    x += `<voice>1</voice><type>${pc.type}</type>` + '<dot/>'.repeat(pc.dots);
    if (pc.tuplet) x += '<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>';
    x += '<notations>';
    if (tieStop) x += '<tied type="stop"/>';
    if (tieStart) x += '<tied type="start"/>';
    x += `<technical><string>${S - string}</string><fret>${fret}</fret></technical></notations></note>`;
    return x;
  };
  const restXml = (pc) => `<note><rest/><duration>${pc.dur}</duration><voice>1</voice><type>${pc.type}</type>${'<dot/>'.repeat(pc.dots)}${pc.tuplet ? '<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>' : ''}</note>`;
  for (let b = 0; b < nBars; b++) {
    L.push(`<measure number="${b + 1}">`);
    if (b === 0) {
      L.push(`<attributes><divisions>${div}</divisions><key><fifths>0</fifths></key><time><beats>${q.beatsPerBar}</beats><beat-type>4</beat-type></time>`);
      L.push(`<clef><sign>TAB</sign><line>5</line></clef><staff-details><staff-lines>${S}</staff-lines>`);
      tuning.forEach((m, i) => { const [st, al] = STEPS[m % 12]; L.push(`<staff-tuning line="${i + 1}"><tuning-step>${st}</tuning-step>${al ? '<tuning-alter>1</tuning-alter>' : ''}<tuning-octave>${Math.floor(m / 12) - 1}</tuning-octave></staff-tuning>`); });
      if (capo) L.push(`<capo>${capo}</capo>`);
      L.push('</staff-details></attributes>');
      L.push(`<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${Math.round(bpm)}</per-minute></metronome></direction-type><sound tempo="${Math.round(bpm)}"/></direction>`);
    }
    let pos = 0;
    const evs = [];
    for (let k = 0; k < spb; k++) { const e = bySlot.get(b * spb + k); if (e && e.notes.some((n) => n.string >= 0)) evs.push(e); }
    for (let i = 0; i < evs.length; i++) {
      const e = evs[i], start = e.slot - b * spb;
      if (start > pos) for (const pc of pieces(start - pos, div)) L.push(restXml(pc));
      const next = i + 1 < evs.length ? evs[i + 1].slot - b * spb : spb;
      const placed = e.notes.filter((n) => n.string >= 0);
      const want = Math.max(...placed.map((n) => n.len || 1));
      const len = Math.max(1, Math.min(want, next - start));
      const ps = pieces(len, div);
      ps.forEach((pc, pi) => placed.forEach((n, ni) => L.push(noteXml(n.pitch, pc, ni > 0, n.string, n.fret, pi < ps.length - 1, pi > 0))));
      pos = start + len;
    }
    if (pos < spb) for (const pc of pieces(spb - pos, div)) L.push(restXml(pc));
    L.push('</measure>');
  }
  L.push('</part></score-partwise>');
  return L.join('\n');
}
