// End-to-end check: synth guitar part → MP3 → decode → model → pipeline. Node only (no browser APIs).
import fs from 'fs';
import { execFileSync } from 'child_process';
import { GraphModel } from '../engine/graph.js';
import { makeWindows, stitch } from '../engine/transcribe.js';
import { renderPluck } from '../engine/synth.js';
import { detectNotes, analyzeRhythm, buildTab, DEFAULTS } from '../engine/pipeline.js';
import { renderAscii, renderChordSheet } from '../engine/tab.js';
import { toMidi, toMusicXML } from '../engine/export.js';
import { chordVoicings } from '../engine/fretboard.js';
import { WIN } from '../engine/constants.js';
import { harmonic } from '../engine/hpss.js';
const HP = process.argv.includes('--drums');

const BPM = 100, spb = 60 / BPM, lead = 0.37;
const truth = [];
const riff = [[0,52],[0.5,55],[1,57],[1.5,59],[2,62],[2.5,59],[3,57],[3.25,55],[3.5,52]];
for (let bar = 0; bar < 4; bar++) for (const [b, p] of riff) truth.push({ beat: bar * 4 + b, pitch: p + (bar === 3 ? 2 : 0), durB: 0.45 });
const chords = { G: [43, 47, 50, 55, 59, 67], C: [48, 52, 55, 60, 64], D: [50, 57, 62, 66], Em: [40, 47, 52, 55, 59, 64] };
const prog = ['G', 'C', 'D', 'Em', 'G', 'C', 'D', 'Em'];
prog.forEach((c, i) => { for (const b of [0, 2]) for (const [k, p] of chords[c].entries()) truth.push({ beat: 16 + i * 4 + b + k * 0.012, pitch: p, durB: 1.9, chord: c }); });
const notes = truth.map((n) => ({ t: lead + n.beat * spb, dur: n.durB * spb, pitch: n.pitch, amp: 0.7 }));
const dur = lead + 48 * spb + 1;
let audio = renderPluck(notes, dur, 44100);
for (let i = 0; i < audio.length; i++) audio[i] += 0.004 * (Math.random() * 2 - 1);
fs.mkdirSync('/tmp/plk', { recursive: true });
fs.writeFileSync('/tmp/plk/in.f32', Buffer.from(audio.buffer));
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'f32le', '-ar', '44100', '-ac', '1', '-i', '/tmp/plk/in.f32', '-b:a', '192k', '/tmp/plk/test.mp3']);
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', '/tmp/plk/test.mp3', '-f', 'f32le', '-ar', '22050', '-ac', '1', '/tmp/plk/dec.f32']);
const db = fs.readFileSync('/tmp/plk/dec.f32');
const a22 = new Float32Array(db.buffer.slice(db.byteOffset, db.byteOffset + db.length));

const json = JSON.parse(fs.readFileSync('../model/model.json'));
const bin = fs.readFileSync('../model/group1-shard1of1.bin');
const m = new GraphModel(json, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));
const src = HP ? harmonic(a22) : a22;
const wins = makeWindows(src);
if (HP) console.log('drum removal ON');
const t0 = Date.now();
const res = wins.map((w) => {
  const o = m.execute(w, [1, WIN, 1]);
  const k = (t, bins) => t.data.slice(15 * bins, 157 * bins);
  return { frames: k(o['Identity_1:0'], 88), onsets: k(o['Identity_2:0'], 88), contours: k(o.contour, 264) };
});
console.log(`model: ${wins.length} windows in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const model = stitch(res, a22.length);
const s = { ...DEFAULTS };
const det = detectNotes(model, s);
// accuracy vs truth (onset within 50 ms, same pitch)
const used = new Set(); let hit = 0, errSum = 0;
for (const tn of notes) {
  let bi = -1, be = 0.05;
  det.forEach((d, i) => { if (!used.has(i) && d.pitch === tn.pitch && Math.abs(d.t - tn.t) < be) { be = Math.abs(d.t - tn.t); bi = i; } });
  if (bi >= 0) { used.add(bi); hit++; errSum += det[bi].t - tn.t; }
}
const fps=det.filter((d,i)=>!used.has(i)); console.log("FP sample:", fps.slice(0,40).map(d=>`${d.pitch}@${d.t.toFixed(2)}a${d.amp.toFixed(2)}d${d.dur.toFixed(2)}`).join(" "));
{const miss=[];for(const tn of notes){if(!det.some(d=>d.pitch===tn.pitch&&Math.abs(d.t-tn.t)<0.05))miss.push(`${tn.pitch}@${tn.t.toFixed(2)}`);}console.log("MISS:",miss.join(" "));}
const P = hit / det.length, R = hit / notes.length;
console.log(`notes: truth ${notes.length} detected ${det.length} precision ${P.toFixed(3)} recall ${R.toFixed(3)} mean onset err ${(1000 * errSum / hit).toFixed(1)} ms`);
const rhythm = analyzeRhythm(model, a22, s, det);
console.log(`tempo ${rhythm.bpm.toFixed(2)} (truth ${BPM}); beats ${rhythm.beats.length}; first beat ${rhythm.beats[0].toFixed(3)} (truth phase ${(lead % spb).toFixed(3)}) autoDownbeat ${rhythm.autoDownbeat}`);
let phaseErr = 0; for (const b of rhythm.beats) { const r = ((b - lead) / spb); phaseErr = Math.max(phaseErr, Math.abs(r - Math.round(r)) * spb); }
console.log(`max beat phase error ${(phaseErr * 1000).toFixed(1)} ms`);
const tab = buildTab(model, det, rhythm, s);
let bad = 0, unplaced = 0, maxSpan = 0;
for (const ev of tab.q.events) {
  const fr = [];
  const strs = new Set();
  for (const n of ev.notes) {
    if (n.string < 0) { unplaced++; continue; }
    if (s.tuning[n.string] + s.capo + n.fret !== n.pitch) bad++;
    if (strs.has(n.string)) bad++; strs.add(n.string);
    if (n.fret > 0) fr.push(n.fret);
  }
  if (fr.length) maxSpan = Math.max(maxSpan, Math.max(...fr) - Math.min(...fr));
}
console.log(`fingering: events ${tab.q.events.length} invalid ${bad} unplaced ${unplaced} max span ${maxSpan}`);
// quantization check for riff: slot positions should be on the 8th grid
const riffEv = tab.q.events.filter((e) => tab.slotTime(e.slot) < lead + 16 * spb - 0.1);
const off8 = riffEv.filter((e) => e.slot % 2 !== 0).length;
console.log(`riff events ${riffEv.length}, off 8th grid: ${off8} (truth has 4 on 16ths)`);
const chordSeq = tab.chords.segments.map((x) => x.label.name).join(' ');
console.log('chords:', chordSeq);
const ascii = renderAscii(tab.q, s.tuning, { chords: tab.marks, width: 78 });
console.log(ascii.split('\n').slice(0, 16).join('\n'));
console.log(renderChordSheet(tab.chords.perBeat, 4, tab.downbeat).split('\n').slice(0, 6).join('\n'));
const mid = toMidi(tab.q, rhythm.bpm); fs.writeFileSync('/tmp/plk/out.mid', mid);
const xml = toMusicXML(tab.q, s.tuning, { bpm: rhythm.bpm }); fs.writeFileSync('/tmp/plk/out.musicxml', xml);
for (const c of ['G', 'C', 'D', 'Em', 'F', 'Bm', 'A7']) {
  const map = { G: [7, [7, 11, 2], 7], C: [0, [0, 4, 7], 0], D: [2, [2, 6, 9], 2], Em: [4, [4, 7, 11], 4], F: [5, [5, 9, 0], 5], Bm: [11, [11, 2, 6], 11], A7: [9, [9, 1, 4, 7], 9] };
  const [r, pcs] = map[c]; const v = chordVoicings(pcs, r, s.tuning, 0, 2);
  console.log(c.padEnd(3), v.map((x) => x.frets.map((f) => (f < 0 ? 'x' : f)).join('')).join('  '));
}
// grid / meter variants: MusicXML measures must sum exactly, every note placed validly
for (const subdiv of [2, 3, 4, 6]) for (const beatsPerBar of [3, 4, 6]) {
  const s2 = { ...s, subdiv, beatsPerBar };
  const r2 = analyzeRhythm(model, a22, s2, det);
  const t2 = buildTab(model, det, r2, s2);
  const x = toMusicXML(t2.q, s2.tuning, { bpm: r2.bpm });
  const ms = x.split('<measure ').slice(1); let bad = 0;
  for (const m of ms) { let sum = 0; for (const n of m.split('<note>').slice(1)) { if (n.startsWith('<chord/>')) continue; sum += +n.match(/<duration>(\d+)<\/duration>/)[1]; } if (sum !== subdiv * beatsPerBar) bad++; }
  const types = new Set([...x.matchAll(/<type>(\w+)<\/type>/g)].map((m) => m[1]));
  console.log(`grid ${subdiv} meter ${beatsPerBar}: measures ${ms.length} bad ${bad} types ${[...types].join(',')}`);
}
// alternate tunings / capo: every placed note must sound its pitch
for (const [name, tuning, capo] of [['DropD', [38, 45, 50, 55, 59, 64], 0], ['capo2', [40, 45, 50, 55, 59, 64], 2], ['7str', [35, 40, 45, 50, 55, 59, 64], 0], ['bass4', [28, 33, 38, 43], 0]]) {
  const s3 = { ...s, tuning, capo };
  const d3 = detectNotes(model, s3), r3 = analyzeRhythm(model, a22, s3, d3), t3 = buildTab(model, d3, r3, s3);
  let ok = 0, bad = 0, un = 0;
  for (const e of t3.q.events) { const used = new Set(); for (const n of e.notes) { if (n.string < 0) { un++; continue; } if (tuning[n.string] + capo + n.fret === n.pitch && !used.has(n.string) && n.fret >= 0) ok++; else bad++; used.add(n.string); } }
  console.log(`${name}: placed ${ok} invalid ${bad} unplaced ${un}`);
}
