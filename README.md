# Plektron v1.2.1

**Audio → guitar tab, entirely in your browser.** Drop in an MP3 (or WAV, OGG, M4A, FLAC) and Plektron listens for the notes, finds the beat, works out a playable fingering, and gives you tab, chords with diagrams, MIDI and MusicXML. Nothing is uploaded and there are no external libraries.

## What to expect

- **Solo or clearly audible guitar** (riffs, lead lines, fingerpicking, strummed chords): good results. Expect to fix some notes by ear.
- **Band mixes**: the note detector can't tell instruments apart, so on its own it would turn bass and vocals into "guitar" notes and drums into stray hits. Plektron first runs AI source separation (Spleeter's 4-stem model, ported to the browser) and keeps only the "other" stem: guitars plus keys/synths, with vocals, drums and bass removed. Heavily distorted guitar is still harder than clean guitar, and keys/synth parts stay in the stem, so expect to edit.
- Every automatic transcription is a draft. The editor lets you delete, add, re-pitch and re-finger notes. The tab synth lets you A/B it against the recording.

### Listen to (Source panel)

- **Guitar & keys (AI)** separates the song with Deezer's Spleeter 4-stem U-Net and keeps the "other" stem. The model (43 MB, mixed int8/float16) downloads once and is cached by the browser, then runs on your device with a WebAssembly SIMD kernel across several workers. A 3-minute song takes roughly 15–60 s depending on the machine. The port matches the original TensorFlow model to a 0.07% mean mask difference, and the separated stem correlates 0.98 with TensorFlow's below 5 kHz.
- **Sides (L−R)** subtracts the right channel from the left. Anything mixed dead-centre (lead vocal, bass, kick, snare) cancels out. Guitars, which are usually panned or double-tracked left and right, stay.
- **Remove drums** runs harmonic/percussive separation (median filtering of the spectrogram) before the note detector.
- **Auto** uses AI separation when the song looks like a band mix: a strong, centred low end (bass/kick), or a very bass-heavy mono file. Otherwise it uses the full mix untouched, which is best for solo guitar.
- Tempo and beats always come from the full mix, since the drums are the best timekeeper.
- **Play what Plektron listens to** plays the isolated signal, so you can hear what the detector is working from. **Export → Isolated audio (.wav)** saves it.
- **Re-analyse with these** reruns detection. Results for each source are cached, so switching back is instant.

## How it works

1. **Decode** the file with the browser's audio decoder and resample to 22.05 kHz. Pick the source (full mix, sides, left, right), and optionally strip drums with harmonic/percussive separation (`engine/hpss.js`, run in a worker).
2. **Note detection.** Plektron runs Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) network over 2-second windows, using a small built-in TensorFlow-graph interpreter (`engine/graph.js`) that runs the model directly. The windows run in parallel across Web Workers. The interpreter's output matches TensorFlow.js to within 10⁻⁴.
3. **Note events.** Peaks in the onset activations become notes. Overtones and ringing re-detections are cleaned up, and polyphony is capped.
4. **Rhythm.** A spectral-flux plus pitched-onset envelope gives the tempo (prior-weighted autocorrelation) and the beats (Ellis dynamic-programming beat tracker). Chord changes and low notes pick the downbeat. Notes are snapped to an 8th, 16th or triplet grid that follows tempo drift.
5. **Fingering.** A Viterbi search over every playable string/fret assignment for each chord or note. The cost function covers hand span, position changes (scaled by time available), preferred position, open strings, skipped strings, and cutting off ringing notes. A weak outlier note can be left out of the tab when it would force an unplayable shape.
6. **Chords.** The note activations give beat-synchronous chroma and bass chroma. These are scored against 108 chord templates and smoothed with an HMM that resists changes. Voicings come from a search over every playable 4-fret window, so they work in any tuning and with a capo.

## Using it

- **Open** a file (or drag it onto the page), or press **Demo** for a synthesized test song.
- **Tab / Roll / Chords** switch views. Tap a note to select it, or tap empty space to seek. Drag to scroll, pinch to zoom, and use two fingers to pan.
- **Edit:** use the ◀ ▶ buttons or the ←/→ keys to move a note to another string. Pitch ± uses ↑/↓, Delete removes a note, and in Roll you double-tap to add one. Ctrl+Z / Ctrl+Shift+Z undo and redo. Edits survive changes to the detection settings.
- **Detection presets:** *Lead / riffs* (max two notes at once), *Strummed*, and *Default*.
- **Rhythm:** auto or locked BPM, ½× / 2× tempo, meter, grid, and downbeat nudge.
- **Guitar:** 11 tunings including 7-string and 4/5-string bass, capo, highest fret, hand span, and preferred position.
- **Confirm changes before applying** (on by default): detection, guitar and rhythm controls wait for **Apply** or **Enter**. Presets and tempo buttons apply right away.
- **Export:** ASCII tab (.txt), MusicXML (opens in MuseScore, Guitar Pro and TuxGuitar with string/fret kept), MIDI, chord sheet, or copy the tab to the clipboard.

## Running locally

The model files are loaded with `fetch` and module workers, so the page has to be served over HTTP, not opened from `file://`:

```
python3 -m http.server 8000
# open http://localhost:8000
```

GitHub Pages works as-is.

## Files

```
index.html, style.css, main.js     UI, state, editing, export wiring
engine/graph.js                    TensorFlow graph-model interpreter (Conv2D, StridedSlice, MirrorPad, …)
engine/model-worker.js             Web Worker running model windows
engine/transcribe.js               decode/resample, source isolation + Auto choice, windowing, worker pool
engine/hpss.js, prep-worker.js     harmonic/percussive separation (drum removal) in a worker
engine/unet.js                     Spleeter 4-stem U-Net (conv, transposed conv, BN, ELU)
engine/simd.js, simd-bytes.js      WebAssembly SIMD vec×mat kernel (source: tools/vecmat.wat)
engine/separate.js, -worker.js     STFT → U-Net masks (worker pool) → masked inverse STFT
tools/                             export_spleeter.py (checkpoint → int8/f16 weights), vecmat.wat + build_wasm.mjs
engine/notes.js                    note events from activations + cleanup
engine/beats.js                    FFT, onset envelope, tempo, beat tracking, downbeat
engine/chords.js                   chord recognition (templates + Viterbi)
engine/fretboard.js                quantization, Viterbi fingering, chord voicings
engine/tab.js, engine/export.js    ASCII tab, chord sheet, MIDI, MusicXML
engine/synth.js, engine/player.js  Karplus–Strong tab synth, Web Audio transport
views/                             tab canvas, piano roll, chord sheet, gestures
model/                             Basic Pitch weights, Spleeter 4-stem weights (spleeter4.json + spleeter4-0/1/2.bin), licenses
tests/                             Node tests (see below)
```

## Tests

- `node tests/e2e.mjs` builds a synthetic guitar part (riff plus strummed G–C–D–Em), runs it through MP3 using `ffmpeg`, then through the full pipeline. It reports note precision and recall, tempo and beat-phase error, fingering validity, MusicXML measure sums for every grid and meter, and alternate tunings. Current results: precision 0.91, recall 0.86, tempo 99.4 against 100 BPM, 0 invalid fingerings, 0 bad measures.
- `node tests/parity.mjs` compares `graph.js` against TensorFlow.js. This one needs `@tensorflow/tfjs` installed, as a dev-only dependency.
- `node tests/resample.test.mjs` checks the fallback resampler.
- `node tests/hpss.test.mjs` checks that drum removal rebuilds steady tones exactly and removes click trains.
- `node tests/e2e.mjs --drums` runs the end-to-end test with drum removal on (shows its cost on clean guitar).
- `node tests/prof.mjs` profiles one model window, op by op.

## Credits and licenses

- Note-detection model: **Basic Pitch** © 2022 Spotify AB, Apache License 2.0 (`model/LICENSE-basic-pitch.txt`).
- Source-separation model: **Spleeter** 4-stem pretrained weights © 2019-present Deezer SA, MIT License (`model/LICENSE-spleeter.txt`). Converted and quantized with `tools/export_spleeter.py`. The note-creation logic in `engine/notes.js` is adapted from Basic Pitch's `toMidi.ts`.
- Everything else is original to Plektron.

## Changelog

**1.2.1**: The separation model is split into `spleeter4-0.bin`, `-1.bin` and `-2.bin` (each under 20 MB) so they can be uploaded through GitHub's web interface, which rejects files over 25 MB. Upload all three alongside `spleeter4.json`.

**1.2.0**: AI source separation. New *Guitar & keys (AI)* source, a browser port of Spleeter's 4-stem U-Net using a hand-written WebAssembly SIMD kernel. It removes vocals, drums and bass before note detection, and Auto uses it for band mixes. Added *Isolated audio (.wav)* export.

**1.1.0**: Band-mix handling. New *Listen to* options (Auto / Full mix / Sides / Left / Right) and *Remove drums* (harmonic/percussive separation). Auto picks Sides with drum removal for band mixes. Beats always come from the full mix. Added *Play what Plektron listens to*, and per-source result caching.

**1.0.0**: First release. Tab, Roll and Chords views, editor with undo, MIDI/MusicXML/TXT export, phone layout with docked view and tabs, and confirm-before-apply mode.

## License

MIT — see [LICENSE](LICENSE).

---

Created with help from Claude AI
