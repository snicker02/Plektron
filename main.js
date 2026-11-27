// Plektron v1.2.1 — audio → guitar tab, entirely in the browser.
import { TUNINGS, SR, midiName } from './engine/constants.js';
import { decodeChannels, runModel, chooseAuto, sourceSignal, mixOf, removeDrums, SOURCES } from './engine/transcribe.js';
import { DEFAULTS, detectNotes, analyzeRhythm, buildTab } from './engine/pipeline.js';
import { onsetEnvelope } from './engine/beats.js';
import { separateOther, SEP_MODEL_MB } from './engine/separate.js';
import { positionsFor } from './engine/fretboard.js';
import { renderAscii, renderChordSheet } from './engine/tab.js';
import { toMidi, toMusicXML } from './engine/export.js';
import { renderPluck } from './engine/synth.js';
import { Player } from './engine/player.js';
import { TabView } from './views/tabview.js';
import { RollView } from './views/rollview.js';
import { ChordView } from './views/chordview.js';

const VERSION = '1.2.1';
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const LS = { get(k, d) { try { const v = localStorage.getItem('plektron.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem('plektron.' + k, JSON.stringify(v)); } catch (e) {} } };

// ---------------- state ----------------
const PRESETS = {
  default: { onsetThresh: 0.5, frameThresh: 0.3, minNoteMs: 70, minAmp: 0.2, maxPoly: 6, overtones: true, melodia: true },
  lead: { onsetThresh: 0.45, frameThresh: 0.32, minNoteMs: 50, minAmp: 0.22, maxPoly: 2, overtones: true, melodia: true },
  chords: { onsetThresh: 0.55, frameThresh: 0.25, minNoteMs: 90, minAmp: 0.15, maxPoly: 6, overtones: false, melodia: true },
};
const S0 = { ...DEFAULTS, tuningName: 'Standard (EADGBE)' };
let s = { ...S0, ...LS.get('settings', {}) };
if (!TUNINGS[s.tuningName]) s.tuningName = S0.tuningName;
s.tuning = TUNINGS[s.tuningName];
let pending = null; // settings awaiting Apply

const st = {
  name: '', original: null, channels: null, mix: null, srcAudio: null, srcLabel: '', auto: null, model: null, env: null, cache: new Map(), analysedKey: '',
  notes: [], rhythm: null, tab: null, fingerOf: new Map(),
  selKey: null, edits: { deleted: [], locks: {}, added: [] }, undo: [], redo: [],
  synthDirty: true, busy: false, token: null, lastUserScroll: 0, view: LS.get('view', 'tab'),
};
const player = new Player();
const app = {
  get selected() { return st.selKey ? { key: st.selKey } : null; },
  select(n) { st.selKey = n ? n.key : null; updateSel(); redraw(); },
  seek(t) { player.seek(t); tick(true); },
  seekSlot(slot) { if (st.tab) app.seek(Math.max(0, st.tab.slotTime(slot))); },
  addNote(pitch, t) { addNote(pitch, t); },
  userScrolled() { st.lastUserScroll = performance.now(); },
  onZoom(z) { $('#zoom').value = z; LS.set('zoom', z); },
};
const tabView = new TabView($('#tabCanvas'), app);
const rollView = new RollView($('#rollCanvas'), app);
const chordView = new ChordView($('#chordView'), app);
tabView.setZoom(LS.get('zoom', 1)); $('#zoom').value = tabView.zoom;

// ---------------- controls ----------------
const tuningSel = $('#tuningSel');
for (const name of Object.keys(TUNINGS)) { const o = document.createElement('option'); o.textContent = name; tuningSel.appendChild(o); }

const fmt = (k, v) => {
  if (k === 'minNoteMs') return `${v}`;
  if (k === 'capo' || k === 'maxFret' || k === 'maxSpan' || k === 'prefFret' || k === 'maxPoly') return `${v}`;
  return (+v).toFixed(2);
};
function readCtl(el) {
  if (el.type === 'checkbox') return el.checked;
  if (el.tagName === 'SELECT' && el.dataset.k === 'tuningName') return el.value;
  return +el.value;
}
function writeControls(src) {
  for (const el of $$('[data-k]')) {
    const k = el.dataset.k, v = src[k];
    if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    const out = el.parentElement.querySelector('output'); if (out) out.textContent = fmt(k, v);
  }
}
writeControls(s);

const confirmEl = $('#confirmMode');
confirmEl.checked = LS.get('confirm', true);
confirmEl.addEventListener('change', () => { LS.set('confirm', confirmEl.checked); if (!confirmEl.checked && pending) applyPending(); });

for (const el of $$('[data-k]')) {
  const ev = el.type === 'range' ? 'input' : 'change';
  el.addEventListener(ev, () => {
    const k = el.dataset.k, v = readCtl(el);
    const out = el.parentElement.querySelector('output'); if (out) out.textContent = fmt(k, v);
    if (confirmEl.checked && st.model) {
      pending = pending || { ...s }; pending[k] = v;
      markPending(el.closest('.sec'));
    } else if (ev === 'change' || !confirmEl.checked) {
      s[k] = v; settingsChanged();
    }
  });
  if (el.type === 'range') el.addEventListener('change', () => { if (!confirmEl.checked || !st.model) { s[el.dataset.k] = readCtl(el); settingsChanged(); } });
}
let pendingSecs = new Set();
function markPending(sec) {
  if (sec) { sec.classList.add('pending'); pendingSecs.add(sec.dataset.sec); }
  $('#applyBtn').hidden = false; $('#applyFloat').hidden = false; syncTabsPending();
}
function clearPending() {
  pending = null; pendingSecs.clear();
  $$('.sec.pending').forEach((x) => x.classList.remove('pending'));
  $('#applyBtn').hidden = true; $('#applyFloat').hidden = true; syncTabsPending();
}
function applyPending() { if (!pending) return; s = { ...pending }; clearPending(); settingsChanged(); }
$('#applyBtn').addEventListener('click', applyPending);
$('#applyFloat').addEventListener('click', applyPending);

let recomputeTimer = 0;
function settingsChanged() {
  s.tuning = TUNINGS[s.tuningName];
  LS.set('settings', strip(s));
  clearTimeout(recomputeTimer); recomputeTimer = setTimeout(recompute, 30);
}
const strip = (o) => { const c = { ...o }; delete c.tuning; return c; };
// explicit actions apply immediately and clear anything pending
function explicit(fn) { if (pending) { s = { ...pending }; } clearPending(); fn(); writeControls(s); settingsChanged(); }
$$('[data-preset]').forEach((b) => b.addEventListener('click', () => explicit(() => Object.assign(s, PRESETS[b.dataset.preset]))));
$('#halfTempo').addEventListener('click', () => explicit(() => { if (s.bpm > 0) s.bpm = Math.round(s.bpm * 5) / 10; else s.tempoScale = Math.max(0.25, s.tempoScale * 0.5); }));
$('#dblTempo').addEventListener('click', () => explicit(() => { if (s.bpm > 0) s.bpm = Math.round(s.bpm * 20) / 10; else s.tempoScale = Math.min(4, s.tempoScale * 2); }));
$('#dbPrev').addEventListener('click', () => explicit(() => { s.downbeatShift = (s.downbeatShift || 0) - 1; }));
$('#dbNext').addEventListener('click', () => explicit(() => { s.downbeatShift = (s.downbeatShift || 0) + 1; }));
$('#zoom').addEventListener('input', (e) => { tabView.setZoom(+e.target.value); LS.set('zoom', tabView.zoom); });

const volO = $('#volOrig'), volS = $('#volSynth');
volO.value = LS.get('volOrig', 0.9); volS.value = LS.get('volSynth', 0);
function volChanged() {
  $('#volOrigV').textContent = Math.round(volO.value * 100) + '%'; $('#volSynthV').textContent = Math.round(volS.value * 100) + '%';
  player.setVolumes(+volO.value, +volS.value); LS.set('volOrig', +volO.value); LS.set('volSynth', +volS.value);
  if (+volS.value > 0) ensureSynth();
}
volO.addEventListener('input', volChanged); volS.addEventListener('input', volChanged); volChanged();

// ---------------- view switching ----------------
function setView(v) {
  st.view = v; LS.set('view', v);
  $$('#viewSeg button').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  $('#tabCanvas').hidden = v !== 'tab'; $('#rollCanvas').hidden = v !== 'roll'; $('#chordView').hidden = v !== 'chords';
  tabView.dirty = rollView.dirty = true; redraw(); tick(true);
}
$$('#viewSeg button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));

// ---------------- phone tabs ----------------
const mq = window.matchMedia('(max-width: 760px), (max-height: 500px) and (orientation: landscape)');
const tabbar = $('#tabbar');
function buildTabs() {
  tabbar.innerHTML = '';
  const secs = $$('#panel > .sec').filter((x) => !x.hidden);
  let cur = LS.get('tab', 'source');
  if (!secs.some((x) => x.dataset.sec === cur)) cur = secs[0] && secs[0].dataset.sec;
  for (const sec of secs) {
    const b = document.createElement('button'); b.textContent = sec.querySelector('h3').textContent; b.dataset.sec = sec.dataset.sec;
    b.addEventListener('click', () => openTab(sec.dataset.sec));
    tabbar.appendChild(b);
  }
  openTab(cur);
}
function openTab(id) {
  LS.set('tab', id);
  $$('#panel > .sec').forEach((x) => x.classList.toggle('tabon', x.dataset.sec === id));
  $$('#tabbar button').forEach((b) => b.classList.toggle('on', b.dataset.sec === id));
  $('#panel').scrollTop = 0;
}
function syncTabsPending() { $$('#tabbar button').forEach((b) => b.classList.toggle('pend', pendingSecs.has(b.dataset.sec))); }
function applyLayout() { document.body.classList.toggle('tabbed', mq.matches); if (mq.matches) buildTabs(); tabView.dirty = rollView.dirty = true; requestAnimationFrame(redraw); }
mq.addEventListener ? mq.addEventListener('change', applyLayout) : mq.addListener(applyLayout);
applyLayout();
if (matchMedia('(pointer: coarse)').matches) $('#editHint').textContent = 'Roll: double-tap empty space to add a note. Drag to scroll, pinch to zoom, two fingers to pan.';

// ---------------- loading ----------------
function status(t) { $('#status').textContent = t; }
function busy(on, title, text) {
  $('#busy').hidden = !on; if (title) $('#busyTitle').textContent = title; if (text != null) $('#busyText').textContent = text;
}
function progress(p) { $('#busyBar').style.width = Math.round(p * 100) + '%'; }

async function loadFile(file) {
  if (st.busy) return;
  try {
    st.busy = true; player.stop();
    busy(true, 'Decoding…', file.name); progress(0);
    const ab = await file.arrayBuffer();
    const { original, channels } = await decodeChannels(ab);
    await analyzeAudio(file.name.replace(/\.[^.]+$/, ''), original, channels);
  } catch (e) {
    console.error(e);
    busy(false); status(e.message === 'cancelled' ? 'Cancelled.' : 'Could not read that file: ' + e.message);
  } finally { st.busy = false; }
}

const srcSel = $('#srcMode'), drumSel = $('#drumMode');
srcSel.value = LS.get('srcMode', 'auto'); drumSel.value = LS.get('drumMode', 'auto');
function wantedKey() {
  const auto = st.auto || { mode: 'mix' };
  const mode = srcSel.value === 'auto' ? auto.mode : srcSel.value;
  const drums = drumSel.value === 'auto' ? mode === 'side' : drumSel.value === 'on';
  return { mode, drums, key: mode + (drums ? '+h' : '') };
}
function syncRelisten() { const b = $('#relisten'); b.disabled = !st.channels || wantedKey().key === st.analysedKey; b.classList.toggle('apply', !b.disabled); }
[srcSel, drumSel].forEach((el) => el.addEventListener('change', () => { LS.set('srcMode', srcSel.value); LS.set('drumMode', drumSel.value); syncRelisten(); }));
$('#relisten').addEventListener('click', async () => {
  if (st.busy || !st.channels) return;
  st.busy = true; player.stop(); $('#playBtn').textContent = '▶';
  try { await listen(); } catch (e) { console.error(e); busy(false); status(e.message === 'cancelled' ? 'Cancelled.' : 'Analysis failed: ' + e.message); }
  finally { st.busy = false; }
});

async function analyzeAudio(name, original, channels) {
  st.name = name; st.original = original; st.channels = channels;
  st.mix = mixOf(channels); st.auto = chooseAuto(channels); st.cache = new Map(); st.analysedKey = '';
  if (!$('#title').value || $('#title').dataset.auto === '1') { $('#title').value = name; $('#title').dataset.auto = '1'; }
  rollView.t0 = 0; tabView.scrollY = 0;
  await listen();
  $('#empty').hidden = true;
}

/** Build the chosen source, optionally strip drums, run the model (cached per source). */
async function listen() {
  const { mode, drums, key } = wantedKey();
  st.token = { cancelled: false };
  const t0 = performance.now();
  let entry = st.cache.get(key);
  if (!entry) {
    let src;
    if (mode === 'ai') {
      let stem = st.cache.get('ai-stem');
      if (!stem) {
        busy(true, 'Separating instruments…', ''); progress(0);
        stem = await separateOther(st.channels, (phase, p) => {
          progress(p);
          $('#busyTitle').textContent = phase === 'download' ? `Downloading separation model (${SEP_MODEL_MB} MB, first time only)…` : phase === 'separate' ? 'Removing vocals, drums and bass…' : 'Rebuilding the guitar track…';
          $('#busyText').textContent = `${Math.round(p * 100)}% · ${((performance.now() - t0) / 1000).toFixed(0)} s`;
        }, st.token);
        st.cache.set('ai-stem', stem);
      }
      src = mixOf(stem);
    } else src = sourceSignal(st.channels, mode);
    if (drums) {
      busy(true, 'Removing drums…', ''); progress(0);
      src = await removeDrums(src, (p) => progress(p), st.token);
    }
    busy(true, 'Listening for notes…', `${(src.length / SR).toFixed(1)} s · ${SOURCES[mode]}${drums ? ', drums removed' : ''}`); progress(0);
    const model = await runModel(src, (p) => { progress(p); $('#busyText').textContent = `${Math.round(p * 100)}% · ${((performance.now() - t0) / 1000).toFixed(0)} s · ${SOURCES[mode]}${drums ? ', drums removed' : ''}`; }, st.token);
    entry = { model, src, env: onsetEnvelope(st.mix, model.onsets, model.T) };
    st.cache.set(key, entry);
  }
  st.model = entry.model; st.srcAudio = entry.src; st.env = entry.env; st.analysedKey = key;
  st.srcLabel = `${SOURCES[mode]}${srcSel.value === 'auto' ? ` (auto: ${st.auto.why})` : ''}${drums ? ' · drums removed' : ''}`;
  st.edits = { deleted: [], locks: {}, added: [] }; st.undo = []; st.redo = []; st.selKey = null;
  setPlaybackSource();
  rollView.setModel(st.model);
  clearPending(); recompute();
  busy(false);
  $('#fileInfo').innerHTML = `${esc(st.name)} · ${fmtTime(st.original.duration)} · analysed in ${((performance.now() - t0) / 1000).toFixed(1)} s<br>Listening to: <b>${esc(st.srcLabel)}</b>`;
  updateUndo(); syncRelisten();
}
function wavBlob(x, sr) {
  const n = x.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  let peak = 1e-9; for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(x[i]));
  const g = peak > 0.99 ? 0.99 / peak : 1;
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-32767, Math.min(32767, Math.round(x[i] * g * 32767))), true);
  return new Blob([buf], { type: 'audio/wav' });
}
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function setPlaybackSource() {
  if (!st.original) return;
  const t = player.time, was = player.playing;
  if ($('#hearSrc').checked && st.srcAudio) {
    const ctx = player.ensure(), b = ctx.createBuffer(1, st.srcAudio.length, SR);
    b.getChannelData(0).set(st.srcAudio); player.setOriginal(b);
  } else player.setOriginal(st.original);
  if (was) player.play(t); else player.seek(t);
}
$('#hearSrc').addEventListener('change', setPlaybackSource);
$('#title').addEventListener('input', (e) => { e.target.dataset.auto = '0'; });
$('#cancelBtn').addEventListener('click', () => { if (st.token) st.token.cancelled = true; });
$('#file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) loadFile(f); e.target.value = ''; });
window.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
window.addEventListener('dragleave', () => document.body.classList.remove('dragging'));
window.addEventListener('drop', (e) => { e.preventDefault(); document.body.classList.remove('dragging'); const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

// Demo: a synthesized riff + chord progression, so the app can be tried without a file.
async function demo() {
  if (st.busy) return;
  st.busy = true;
  try {
    const bpm = 100, b = 60 / bpm, lead = 0.4, notes = [];
    const riff = [[0, 52], [0.5, 55], [1, 57], [1.5, 59], [2, 62], [2.5, 59], [3, 57], [3.25, 55], [3.5, 52]];
    for (let bar = 0; bar < 4; bar++) for (const [x, p] of riff) notes.push({ t: lead + (bar * 4 + x) * b, dur: 0.45 * b, pitch: p + (bar === 3 ? 2 : 0), amp: 0.7 });
    const ch = { G: [43, 47, 50, 55, 59, 67], C: [48, 52, 55, 60, 64], D: [50, 57, 62, 66], Em: [40, 47, 52, 55, 59, 64] };
    ['G', 'C', 'D', 'Em', 'G', 'C', 'D', 'Em'].forEach((c, i) => { for (const x of [0, 2]) ch[c].forEach((p, k) => notes.push({ t: lead + (16 + i * 4 + x + k * 0.012) * b, dur: 1.9 * b, pitch: p, amp: 0.7 })); });
    const audio = renderPluck(notes, lead + 48 * b + 1, SR, 0.06);
    const ctx = player.ensure();
    const buf = ctx.createBuffer(1, audio.length, SR); buf.getChannelData(0).set(audio);
    await analyzeAudio('Demo riff', buf, [audio]);
  } catch (e) { console.error(e); busy(false); status(e.message === 'cancelled' ? 'Cancelled.' : 'Demo failed: ' + e.message); }
  finally { st.busy = false; }
}
$('#demoBtn').addEventListener('click', demo); $('#demoBtn2').addEventListener('click', demo);

// ---------------- analysis ----------------
const keyOf = (n) => `${n.pitch}@${Math.round(n.t * 100)}`;
function applyEdits(det) {
  const del = new Set(st.edits.deleted), out = [];
  for (const n of det) {
    n.key = keyOf(n);
    if (del.has(n.key)) continue;
    const lk = st.edits.locks[n.key];
    if (lk != null) { n.lockString = lk; n.locked = true; }
    out.push(n);
  }
  for (const a of st.edits.added) {
    if (del.has(a.key)) continue;
    const n = { ...a, added: true }; const lk = st.edits.locks[n.key];
    if (lk != null) { n.lockString = lk; n.locked = true; }
    out.push(n);
  }
  return out.sort((x, y) => x.t - y.t || x.pitch - y.pitch);
}

function recompute() {
  if (!st.model) return;
  const t0 = performance.now();
  s.tuning = TUNINGS[s.tuningName];
  const det = detectNotes(st.model, s);
  st.notes = applyEdits(det);
  st.rhythm = analyzeRhythm(st.model, st.mix, s, st.notes, st.env);
  st.tab = buildTab(st.model, st.notes, st.rhythm, s);
  st.fingerOf = new Map();
  for (const ev of st.tab.q.events) for (const n of ev.notes) st.fingerOf.set(n.key, { string: n.string, fret: n.fret, slot: ev.slot });
  publish();
  st.synthDirty = true; if (+volS.value > 0) ensureSynth();
  const placed = [...st.fingerOf.values()].filter((f) => f.string >= 0).length;
  status(`${st.notes.length} notes · ${placed} in tab · ${st.rhythm.bpm.toFixed(1)} BPM · ${(performance.now() - t0).toFixed(0)} ms`);
  $('#tempoInfo').textContent = `${s.bpm > 0 ? 'locked' : 'detected'} ${st.rhythm.bpm.toFixed(1)} BPM`;
  updateSel();
}

function publish() {
  const tb = st.tab, q = tb.q;
  const isBarBeat = (i) => ((i * q.subdiv - q.origin) % q.slotsPerBar + q.slotsPerBar) % q.slotsPerBar === 0;
  const barOfBeat = (i) => Math.floor((i * q.subdiv - q.origin) / q.slotsPerBar) + 1;
  tabView.setData({ q, tuning: s.tuning, capo: s.capo, marks: tb.marks });
  rollView.setData({ notes: st.notes, fingerOf: st.fingerOf, tuning: s.tuning, capo: s.capo, maxFret: s.maxFret, beats: st.rhythm.beats, beatsPerBar: s.beatsPerBar, downbeat: tb.downbeat, isBarBeat, barOfBeat, duration: st.original.duration });
  chordView.setData({ segments: tb.chords.segments, perBeat: tb.chords.perBeat, beats: st.rhythm.beats, beatsPerBar: s.beatsPerBar, downbeat: firstBarBeat(q), tuning: s.tuning, capo: s.capo });
}
// beat index (mod bar) at which bars start, consistent with the tab's bar lines
function firstBarBeat(q) { const bpb = q.beatsPerBar; return (((q.origin / q.subdiv) % bpb) + bpb) % bpb; }

function ensureSynth() {
  if (!st.synthDirty || !st.model) return;
  const placed = st.notes.filter((n) => { const f = st.fingerOf.get(n.key); return f && f.string >= 0; });
  player.setSynth(renderPluck(placed, st.original.duration, SR), SR);
  st.synthDirty = false;
}

// ---------------- editing ----------------
function snapshot() { st.undo.push(JSON.stringify(st.edits)); if (st.undo.length > 200) st.undo.shift(); st.redo = []; updateUndo(); }
function updateUndo() { $('#undoBtn').disabled = !st.undo.length; $('#redoBtn').disabled = !st.redo.length; }
function undo() { if (!st.undo.length) return; st.redo.push(JSON.stringify(st.edits)); st.edits = JSON.parse(st.undo.pop()); updateUndo(); recompute(); }
function redo() { if (!st.redo.length) return; st.undo.push(JSON.stringify(st.edits)); st.edits = JSON.parse(st.redo.pop()); updateUndo(); recompute(); }
$('#undoBtn').addEventListener('click', undo); $('#redoBtn').addEventListener('click', redo);
$('#clearEdits').addEventListener('click', () => { if (!st.edits.deleted.length && !st.edits.added.length && !Object.keys(st.edits.locks).length) return; snapshot(); st.edits = { deleted: [], locks: {}, added: [] }; recompute(); });

function selNote() { return st.selKey ? st.notes.find((n) => n.key === st.selKey) : null; }
function updateSel() {
  const n = selNote(), el = $('#selInfo');
  const dis = !n;
  ['#altPrev', '#altNext', '#pitchDown', '#pitchUp', '#delNote'].forEach((id) => { $(id).disabled = dis; });
  if (!n) { el.textContent = st.model ? 'Tap a note in Tab or Roll to select it.' : 'Load a song first.'; return; }
  const f = st.fingerOf.get(n.key);
  const where = f && f.string >= 0 ? `string ${s.tuning.length - f.string}, fret ${f.fret}${s.capo ? ` (capo ${s.capo})` : ''}` : 'not placed in tab (out of range or unplayable)';
  const alts = positionsFor(n.pitch, s.tuning, s.capo, s.maxFret).length;
  el.innerHTML = `<b>${midiName(n.pitch)}</b> at ${fmtTime(n.t, true)} · ${Math.round(n.dur * 1000)} ms · ${where}${n.locked ? ' · <span style="color:#4fc3c9">position locked</span>' : ''}${alts > 1 ? ` · ${alts} positions` : ''}`;
  if (alts <= 1) { $('#altPrev').disabled = $('#altNext').disabled = true; }
}
function cyclePosition(dir) {
  const n = selNote(); if (!n) return;
  const opts = positionsFor(n.pitch, s.tuning, s.capo, s.maxFret); if (opts.length < 2) return;
  const f = st.fingerOf.get(n.key);
  let i = opts.findIndex((o) => f && o.string === f.string); if (i < 0) i = 0; else i = (i + dir + opts.length) % opts.length;
  snapshot(); st.edits.locks[n.key] = opts[i].string; recompute();
}
function deleteSel() {
  const n = selNote(); if (!n) return;
  snapshot();
  if (n.added) st.edits.added = st.edits.added.filter((a) => a.key !== n.key); else st.edits.deleted.push(n.key);
  delete st.edits.locks[n.key];
  st.selKey = null; recompute();
}
function shiftPitch(d) {
  const n = selNote(); if (!n) return;
  const p = n.pitch + d; if (p < 21 || p > 108) return;
  snapshot();
  if (n.added) st.edits.added = st.edits.added.filter((a) => a.key !== n.key); else st.edits.deleted.push(n.key);
  delete st.edits.locks[n.key];
  const a = { t: n.t, dur: n.dur, pitch: p, amp: n.amp, onset: true }; a.key = 'a' + keyOf(a);
  st.edits.added.push(a); st.selKey = a.key; recompute();
}
function addNote(pitch, t) {
  if (!st.tab) return;
  const map = st.tab.map, sub = s.subdiv;
  const pos = Math.round(map.toPos(t) * sub) / sub;
  const t0 = Math.max(0, map.toTime(pos)), t1 = map.toTime(pos + 1 / sub * Math.max(1, sub / 2));
  const a = { t: t0, dur: Math.max(0.08, t1 - t0), pitch, amp: 0.6, onset: true }; a.key = 'a' + keyOf(a);
  snapshot(); st.edits.added.push(a); st.selKey = a.key; recompute();
}
$('#altPrev').addEventListener('click', () => cyclePosition(-1));
$('#altNext').addEventListener('click', () => cyclePosition(1));
$('#delNote').addEventListener('click', deleteSel);
$('#pitchDown').addEventListener('click', () => shiftPitch(-1));
$('#pitchUp').addEventListener('click', () => shiftPitch(1));

// ---------------- export ----------------
function header() {
  const lines = [];
  lines.push(($('#title').value || st.name || 'Untitled') + ' — transcribed with Plektron');
  lines.push(`Tuning: ${s.tuningName}${s.capo ? `   Capo: ${s.capo}` : ''}   Tempo: ~${st.rhythm.bpm.toFixed(0)} BPM   Time: ${s.beatsPerBar}/4`);
  lines.push('Automatic transcription — check by ear.');
  return lines;
}
function download(name, data, type) {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
const safe = () => ($('#title').value || st.name || 'tab').replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '_') || 'tab';
function asciiTab() { return renderAscii(st.tab.q, s.tuning, { chords: st.tab.marks, width: 78, header: header() }); }
$$('[data-export]').forEach((b) => b.addEventListener('click', async () => {
  if (!st.tab) { status('Load a song first.'); return; }
  const k = b.dataset.export;
  if (k === 'txt') download(safe() + '_tab.txt', asciiTab(), 'text/plain');
  else if (k === 'mid') download(safe() + '.mid', toMidi(st.tab.q, st.rhythm.bpm), 'audio/midi');
  else if (k === 'musicxml') download(safe() + '.musicxml', toMusicXML(st.tab.q, s.tuning, { capo: s.capo, title: $('#title').value || st.name, bpm: st.rhythm.bpm }), 'application/vnd.recordare.musicxml+xml');
  else if (k === 'chords') download(safe() + '_chords.txt', header().join('\n') + '\n\n' + renderChordSheet(st.tab.chords.perBeat, s.beatsPerBar, firstBarBeat(st.tab.q)), 'text/plain');
  else if (k === 'wav') download(safe() + '_isolated.wav', wavBlob(st.srcAudio, SR), 'audio/wav');
  else if (k === 'copy') { try { await navigator.clipboard.writeText(asciiTab()); status('Tab copied to clipboard.'); } catch (e) { status('Clipboard not available here.'); } }
}));

// ---------------- transport ----------------
const fmtTime = (t, frac) => { const m = Math.floor(t / 60), x = t - m * 60; return `${m}:${frac ? x.toFixed(2).padStart(5, '0') : String(Math.floor(x)).padStart(2, '0')}`; };
function togglePlay() {
  if (!st.original) return;
  if (player.playing) player.pause(); else { if (+volS.value > 0) ensureSynth(); player.play(); }
  $('#playBtn').textContent = player.playing ? '❚❚' : '▶';
}
player.onEnd = () => { $('#playBtn').textContent = '▶'; };
$('#playBtn').addEventListener('click', togglePlay);
$('#stopBtn').addEventListener('click', () => { player.stop(); $('#playBtn').textContent = '▶'; tick(true); });
const seekEl = $('#seek');
seekEl.addEventListener('input', () => { if (st.original) { player.seek((+seekEl.value / 1000) * st.original.duration); tick(true); } });

function redraw() {
  if (st.view === 'tab') tabView.draw(); else if (st.view === 'roll') rollView.draw();
}
let lastT = -1;
function tick(force) {
  const t = player.time;
  if (st.tab && (force || player.playing || t !== lastT)) {
    lastT = t;
    const dur = st.original ? st.original.duration : 0;
    $('#clock').textContent = `${fmtTime(t, true)} / ${fmtTime(dur)}`;
    if (document.activeElement !== seekEl) seekEl.value = dur ? Math.round((t / dur) * 1000) : 0;
    const slot = st.tab.map.toPos(t) * st.tab.q.subdiv - st.tab.q.origin;
    const follow = $('#follow').checked && player.playing && performance.now() - st.lastUserScroll > 2500;
    tabView.setPlayhead(Math.max(0, slot)); rollView.setPlayhead(t);
    if (follow) { if (st.view === 'tab') tabView.ensureVisible(Math.floor(slot)); else if (st.view === 'roll') rollView.ensureVisible(t); }
    if (st.view === 'chords') chordView.setTime(t, follow || force);
  }
  redraw();
}
(function loop() { tick(false); requestAnimationFrame(loop); })();

// ---------------- keyboard ----------------
window.addEventListener('keydown', (e) => {
  const tg = e.target, tag = tg.tagName;
  const textual = tag === 'INPUT' && (tg.type === 'text' || tg.type === 'number') || tag === 'TEXTAREA' || tag === 'SELECT';
  if (e.key === 'Enter') { if (pending && !textual && tag !== 'BUTTON') { e.preventDefault(); applyPending(); } return; }
  if (textual) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
  if (e.key === ' ') { e.preventDefault(); togglePlay(); return; }
  if (!st.selKey) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); cyclePosition(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); cyclePosition(1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); shiftPitch(1); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); shiftPitch(-1); }
  else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); deleteSel(); }
  else if (e.key === 'Escape') app.select(null);
});

setView(st.view);
updateSel();
if (location.protocol === 'file:') status('Open this page through a web server (e.g. GitHub Pages) — browsers block the model files on file://.');
window.__plektron = { st, get s() { return s; }, recompute, demo, VERSION };
