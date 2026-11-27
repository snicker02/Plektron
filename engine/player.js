// Web Audio transport: original recording + optional synth rendering, sample-accurate clock.
export class Player {
  constructor() { this.ctx = null; this.orig = null; this.synth = null; this.srcs = []; this.playing = false; this.offset = 0; this.startAt = 0; this.duration = 0; this.vOrig = 0.9; this.vSynth = 0; this.onEnd = null; }
  ensure() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext; this.ctx = new AC();
      this.gO = this.ctx.createGain(); this.gS = this.ctx.createGain();
      this.gO.connect(this.ctx.destination); this.gS.connect(this.ctx.destination);
      this.gO.gain.value = this.vOrig; this.gS.gain.value = this.vSynth;
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }
  setOriginal(buf) { this.stop(); this.orig = buf; this.duration = buf ? buf.duration : 0; this.offset = 0; }
  setSynth(samples, sr) {
    const was = this.playing, t = this.time; if (was) this._halt();
    if (samples) { const ctx = this.ensure(); const b = ctx.createBuffer(1, samples.length, sr); b.copyToChannel ? b.copyToChannel(samples, 0) : b.getChannelData(0).set(samples); this.synth = b; } else this.synth = null;
    if (was) this.play(t);
  }
  setVolumes(o, s) { this.vOrig = o; this.vSynth = s; if (this.ctx) { this.gO.gain.value = o; this.gS.gain.value = s; } }
  get time() { return this.playing ? Math.min(this.duration, this.offset + this.ctx.currentTime - this.startAt) : this.offset; }
  play(from) {
    if (!this.orig) return;
    const ctx = this.ensure();
    this._halt();
    this.offset = Math.max(0, Math.min(this.duration - 0.01, from ?? this.offset));
    this.startAt = ctx.currentTime + 0.03;
    const mk = (buf, g) => { const s = ctx.createBufferSource(); s.buffer = buf; s.connect(g); s.start(this.startAt, Math.min(this.offset, buf.duration)); return s; };
    this.srcs = [mk(this.orig, this.gO)];
    if (this.synth) this.srcs.push(mk(this.synth, this.gS));
    this.srcs[0].onended = () => { if (this.playing && this.time >= this.duration - 0.05) { this.playing = false; this.offset = 0; this.onEnd && this.onEnd(); } };
    this.playing = true;
  }
  _halt() { for (const s of this.srcs) { try { s.onended = null; s.stop(); } catch (e) {} } this.srcs = []; }
  pause() { if (!this.playing) return; this.offset = this.time; this._halt(); this.playing = false; }
  stop() { this._halt(); this.playing = false; this.offset = 0; }
  seek(t) { if (this.playing) this.play(t); else this.offset = Math.max(0, Math.min(this.duration, t)); }
}
