// Pure analysis pipeline: posteriors → notes → beats → quantized, fingered tab + chords.
import { FPS } from './constants.js';
import { outputToNotes, notesToTime, cleanNotes } from './notes.js';
import { onsetEnvelope, estimateTempo, trackBeats, fixedBeats, chooseDownbeat, makeBeatMap } from './beats.js';
import { quantize, assignFingering } from './fretboard.js';
import { recognizeChords } from './chords.js';
import { chordMarks } from './tab.js';

export const DEFAULTS = {
  onsetThresh: 0.5, frameThresh: 0.3, minNoteMs: 70, minAmp: 0.2, maxPoly: 6, melodia: true, overtones: true,
  tuning: [40, 45, 50, 55, 59, 64], capo: 0, maxFret: 22, maxSpan: 4, prefFret: 3,
  bpm: 0, tempoScale: 1, beatsPerBar: 4, subdiv: 4, downbeatShift: 0, changePenalty: 0.18,
};

/** Stage 1 (cached per file): beat grid. */
export function analyzeRhythm(model, audio, s, notesForDownbeat, envCache) {
  const env = envCache || onsetEnvelope(audio, model.onsets, model.T);
  let bpm = s.bpm > 0 ? s.bpm : estimateTempo(env, FPS) * s.tempoScale;
  let beats;
  if (s.bpm > 0) {
    // phase: align locked tempo with the strongest envelope comb
    const p = 60 / bpm; let best = 0, bs = -1;
    for (let k = 0; k < 40; k++) {
      const off = (k / 40) * p; let sc = 0;
      for (let t = off; t < model.T / FPS; t += p) { const f = Math.round(t * FPS); if (f < env.length) sc += env[f]; }
      if (sc > bs) { bs = sc; best = off; }
    }
    beats = fixedBeats(audio.length / 22050, bpm, best);
  } else beats = trackBeats(env, FPS, bpm);
  if (beats.length > 2) bpm = 60 / median(beats.slice(1).map((b, i) => b - beats[i]));
  const chords = recognizeChords(model.frames, model.frameTimes, beats, { changePenalty: s.changePenalty });
  const changes = chords.perBeat.map((L, i) => (i > 0 && L !== chords.perBeat[i - 1] ? 1 : 0));
  const auto = chooseDownbeat(beats, s.beatsPerBar, (notesForDownbeat || []).slice().sort((a, b) => a.t - b.t), env, FPS, changes);
  return { env, beats, bpm, autoDownbeat: auto, chords };
}

function median(a) { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : 0.5; }

/** Stage 0: notes from posteriors. */
export function detectNotes(model, s) {
  const minMidi = s.tuning[0] + s.capo;
  const maxMidi = s.tuning[s.tuning.length - 1] + s.capo + s.maxFret;
  const raw = outputToNotes(model.frames, model.onsets, model.T, {
    onsetThresh: s.onsetThresh, frameThresh: s.frameThresh,
    minNoteLen: Math.max(1, Math.round((s.minNoteMs / 1000) * FPS)),
    melodia: s.melodia, minMidi, maxMidi,
  });
  return cleanNotes(notesToTime(raw, model.frameTimes), { minAmp: s.minAmp, maxPoly: s.maxPoly, overtones: s.overtones });
}

/** Stage 2: quantize + fingering + chords, given notes and rhythm. */
export function buildTab(model, notes, rhythm, s) {
  const map = makeBeatMap(rhythm.beats);
  const downbeat = ((rhythm.autoDownbeat + s.downbeatShift) % s.beatsPerBar + s.beatsPerBar) % s.beatsPerBar;
  const q = quantize(notes, map, { subdiv: s.subdiv, beatsPerBar: s.beatsPerBar, downbeat });
  assignFingering(q.events, { tuning: s.tuning, capo: s.capo, maxFret: s.maxFret, maxSpan: s.maxSpan, prefFret: s.prefFret });
  const ch = rhythm.chords;
  const marks = chordMarks(ch.segments, q);
  // slot → time for playback sync
  const slotTime = (slot) => map.toTime((slot + q.origin) / q.subdiv);
  return { q, chords: ch, marks, downbeat, slotTime, map };
}
