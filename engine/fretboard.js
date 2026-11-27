// Rhythm quantization, playable fingering by Viterbi search, and chord-voicing search.

/**
 * Quantize notes onto a beat grid and group simultaneous notes into events.
 * @returns {{events, origin, slotsPerBar, subdiv, beatsPerBar}}
 */
export function quantize(notes, beatMap, { subdiv = 4, beatsPerBar = 4, downbeat = 0 } = {}) {
  const q = notes.map((n) => {
    const s = Math.round(beatMap.toPos(n.t) * subdiv);
    const e = Math.round(beatMap.toPos(n.t + n.dur) * subdiv);
    return { n, slot: s, len: Math.max(1, e - s) };
  });
  const minSlot = q.length ? Math.min(...q.map((x) => x.slot)) : 0;
  const bar = beatsPerBar * subdiv;
  let origin = downbeat * subdiv;
  if (minSlot < origin) origin -= bar * Math.ceil((origin - minSlot) / bar);
  else origin += bar * Math.floor((minSlot - origin) / bar);
  const bySlot = new Map();
  for (const x of q) {
    const rel = x.slot - origin;
    if (!bySlot.has(rel)) bySlot.set(rel, []);
    const list = bySlot.get(rel);
    const dup = list.find((y) => y.n.pitch === x.n.pitch);
    if (dup) { if (x.n.amp > dup.n.amp) dup.n = x.n; dup.len = Math.max(dup.len, x.len); continue; }
    list.push(x);
  }
  const events = [...bySlot.entries()].sort((a, b) => a[0] - b[0]).map(([slot, list]) => ({
    slot,
    notes: list.map((x) => ({ ...x.n, len: x.len, string: -1, fret: -1 })).sort((a, b) => b.pitch - a.pitch),
  }));
  return { events, origin, slotsPerBar: bar, subdiv, beatsPerBar };
}

// ---------- fingering ----------
function candidates(notes, tuning, capo, maxFret, maxSpan) {
  const S = tuning.length, out = [];
  const opts = notes.map((n) => {
    const o = [];
    for (let s = 0; s < S; s++) {
      if (n.lockString != null && n.lockString !== s) continue;
      const f = n.pitch - tuning[s] - capo;
      if (f >= 0 && f <= maxFret) o.push([s, f]);
    }
    return o;
  });
  const used = new Array(S).fill(false), cur = [];
  const rec = (i) => {
    if (out.length > 400) return;
    if (i === notes.length) {
      const fr = cur.filter((x) => x[1] > 0).map((x) => x[1]);
      const span = fr.length ? Math.max(...fr) - Math.min(...fr) : 0;
      if (span > maxSpan) return;
      out.push(cur.map((x) => x.slice()));
      return;
    }
    for (const [s, f] of opts[i]) { if (used[s]) continue; used[s] = true; cur.push([s, f]); rec(i + 1); cur.pop(); used[s] = false; }
  };
  rec(0);
  return out;
}

function staticCost(c, prefFret, w) {
  c = c.filter(Boolean);
  const fr = c.filter((x) => x[1] > 0).map((x) => x[1]);
  const span = fr.length ? Math.max(...fr) - Math.min(...fr) : 0;
  const lo = fr.length ? Math.min(...fr) : 0;
  const opens = c.length - fr.length;
  const strs = c.map((x) => x[0]);
  const strSpan = strs.length > 1 ? Math.max(...strs) - Math.min(...strs) + 1 - strs.length : 0; // skipped strings
  return w.span * span * span * 0.15 + w.height * Math.abs(lo - prefFret) * 0.08 - w.open * opens * 0.25 + strSpan * 0.2 + (lo > 12 ? (lo - 12) * 0.1 : 0);
}
function handPos(c) { c = c.filter(Boolean); const fr = c.filter((x) => x[1] > 0).map((x) => x[1]); if (!fr.length) return null; return (Math.min(...fr) + Math.max(...fr)) / 2; }

/**
 * Assign string/fret to every note via Viterbi over per-event fingering candidates.
 * Mutates notes in place (string index into tuning, 0 = lowest string; fret relative to capo).
 * Notes that cannot be placed get string = -1.
 */
export function assignFingering(events, { tuning, capo = 0, maxFret = 22, maxSpan = 4, prefFret = 3, weights = {} } = {}) {
  const w = { span: 1, height: 1, open: 1, move: 1, ring: 1, ...weights };
  const E = events.length; if (!E) return;
  const C = [], SC = [], HP = [];
  for (const ev of events) {
    let notes = ev.notes.filter((n) => n.pitch - (tuning[0] + capo) >= 0 && n.pitch - (tuning[tuning.length - 1] + capo) <= maxFret);
    for (const n of ev.notes) if (!notes.includes(n)) { n.string = -1; n.fret = -1; }
    let cands = [];
    let ns = notes.slice().sort((a, b) => b.amp - a.amp);
    // shrink until playable: widen span first, then drop weakest notes
    while (ns.length) {
      cands = candidates(ns, tuning, capo, maxFret, maxSpan);
      if (!cands.length) cands = candidates(ns, tuning, capo, maxFret, maxSpan + 2);
      if (cands.length) break;
      const dropped = ns.pop(); dropped.string = -1; dropped.fret = -1;
    }
    ev._placed = ns;
    if (!cands.length) cands = [[]];
    let sc = cands.map((c) => staticCost(c, prefFret, w));
    // alternatives that leave out one weak note (outliers / overtones that force bad shapes)
    if (ns.length >= 2) {
      for (let d = 0; d < ns.length; d++) {
        if (ns[d].amp > 0.5 || ns[d].lockString != null) continue;
        const sub = ns.filter((_, i) => i !== d);
        for (const c of candidates(sub, tuning, capo, maxFret, maxSpan)) {
          const full = []; let k = 0;
          for (let i = 0; i < ns.length; i++) full.push(i === d ? null : c[k++]);
          cands.push(full); sc.push(staticCost(c, prefFret, w) + 0.6 + 2.5 * ns[d].amp);
        }
      }
    }
    // keep best 60
    const idx = sc.map((_, i) => i).sort((a, b) => sc[a] - sc[b]).slice(0, 60);
    C.push(idx.map((i) => cands[i])); SC.push(idx.map((i) => sc[i])); HP.push(idx.map((i) => handPos(cands[i])));
  }
  const cost = [Float64Array.from(SC[0])], back = [null];
  for (let e = 1; e < E; e++) {
    const gap = Math.max(1, events[e].slot - events[e - 1].slot) / 4; // in beats-ish
    const prevEv = events[e - 1];
    const K = C[e].length, Kp = C[e - 1].length;
    const ce = new Float64Array(K), be = new Int32Array(K);
    for (let k = 0; k < K; k++) {
      let best = Infinity, bi = 0; const hb = HP[e][k];
      for (let j = 0; j < Kp; j++) {
        let t = cost[e - 1][j];
        const ha = HP[e - 1][j];
        if (ha != null && hb != null) { const d = Math.abs(ha - hb); t += w.move * (d > 1 ? (d * 0.35) / Math.sqrt(gap) + 0.15 : d * 0.08); }
        // cutting off a ringing note on the same string
        const pa = C[e - 1][j];
        for (let i = 0; i < pa.length; i++) {
          if (!pa[i]) continue;
          const nA = prevEv._placed[i];
          if (nA && prevEv.slot + nA.len > events[e].slot) {
            for (const x of C[e][k]) if (x && x[0] === pa[i][0]) { t += w.ring * 0.08; break; }
          }
        }
        if (t < best) { best = t; bi = j; }
      }
      ce[k] = best + SC[e][k]; be[k] = bi;
    }
    cost.push(ce); back.push(be);
  }
  let k = 0; for (let i = 1; i < cost[E - 1].length; i++) if (cost[E - 1][i] < cost[E - 1][k]) k = i;
  for (let e = E - 1; e >= 0; e--) {
    const c = C[e][k];
    events[e]._placed.forEach((n, i) => { if (c[i]) { n.string = c[i][0]; n.fret = c[i][1]; } else { n.string = -1; n.fret = -1; } });
    delete events[e]._placed;
    if (e > 0) k = back[e][k];
  }
}

/** All alternative positions for one pitch (for the editor). */
export function positionsFor(pitch, tuning, capo, maxFret) {
  const o = [];
  for (let s = 0; s < tuning.length; s++) { const f = pitch - tuning[s] - capo; if (f >= 0 && f <= maxFret) o.push({ string: s, fret: f }); }
  return o;
}

// ---------- chord voicings ----------
const voicingCache = new Map();
/**
 * Find playable voicings for a chord. Returns up to `count` shapes:
 * { frets: number[] (per string low→high, -1 = muted, relative to capo), base }
 */
export function chordVoicings(pcs, root, tuning, capo = 0, count = 3) {
  const key = pcs.join(',') + '|' + root + '|' + tuning.join(',') + '|' + capo;
  if (voicingCache.has(key)) return voicingCache.get(key);
  const S = tuning.length, results = [];
  const need = new Set(pcs);
  const optional = pcs.length >= 4 ? (root + 7) % 12 : -1; // fifth may be omitted in 4-note chords
  for (let w = 0; w <= 10; w++) {
    const choices = [];
    for (let s = 0; s < S; s++) {
      const o = [-1];
      const open = (tuning[s] + capo) % 12;
      if (need.has(open)) o.push(0);
      for (let f = Math.max(1, w); f <= w + 3; f++) if (need.has((tuning[s] + capo + f) % 12)) o.push(f);
      choices.push(o);
    }
    const cur = new Array(S);
    const rec = (s) => {
      if (s === S) {
        const sounding = []; for (let i = 0; i < S; i++) if (cur[i] >= 0) sounding.push(i);
        if (sounding.length < Math.min(3, S)) return;
        const bassS = sounding[0];
        if ((tuning[bassS] + capo + cur[bassS]) % 12 !== root) return;
        const have = new Set(sounding.map((i) => (tuning[i] + capo + cur[i]) % 12));
        for (const p of need) if (!have.has(p) && p !== optional) return;
        const fr = sounding.filter((i) => cur[i] > 0).map((i) => cur[i]);
        if (fr.length && Math.max(...fr) - Math.min(...fr) > 3) return;
        const minF = fr.length ? Math.min(...fr) : 0;
        // only use a barre when four fingers are not enough
        let barre = false, fingers = fr.length;
        if (fingers > 4) {
          const atMin = sounding.filter((i) => cur[i] === minF && minF > 0).length;
          if (atMin < 2) return;
          const first = sounding.find((i) => cur[i] === minF);
          for (let i = first; i < S; i++) if (cur[i] === 0 || (cur[i] > 0 && cur[i] < minF)) return; // barre covers these strings
          barre = true; fingers = fr.length - (atMin - 1);
          if (fingers > 4) return;
        }
        let interior = 0; for (let i = bassS; i < S; i++) if (cur[i] < 0) interior++;
        const opens = sounding.length - fr.length;
        const fspan = fr.length ? Math.max(...fr) - Math.min(...fr) : 0;
        const score = sounding.length * 0.4 - interior * 1.5 - minF * 0.25 + opens * 0.15 - (barre ? 0.2 : 0) - fingers * 0.15 - fspan * 0.15;
        const base = fr.length ? Math.min(...fr) : 1;
        results.push({ frets: cur.slice(), score, base });
        return;
      }
      for (const f of choices[s]) { cur[s] = f; rec(s + 1); }
    };
    rec(0);
  }
  const seen = new Set();
  const best = results.sort((a, b) => b.score - a.score).filter((r) => { const k = r.frets.join(','); if (seen.has(k)) return false; seen.add(k); return true; }).slice(0, count);
  voicingCache.set(key, best);
  return best;
}
