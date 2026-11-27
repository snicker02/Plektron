// Basic Pitch framing constants
export const SR = 22050;
export const FFT_HOP = 256;
export const WIN = SR * 2 - FFT_HOP;          // 43844 samples per model window
export const OVERLAP_FRAMES = 30;
export const HALF_OVERLAP = OVERLAP_FRAMES / 2; // 15
export const OVERLAP_SAMPLES = OVERLAP_FRAMES * FFT_HOP; // 7680
export const HOP = WIN - OVERLAP_SAMPLES;     // 36164
export const WIN_FRAMES = 172;
export const KEEP_FRAMES = WIN_FRAMES - OVERLAP_FRAMES; // 142
export const N_PITCH = 88;
export const N_CONTOUR = 264;
export const MIDI_OFFSET = 21;
export const FPS = SR / FFT_HOP;

export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
export const NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
export const midiName = (m) => NOTE_NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);

// Tunings listed low string first (string 6 → string 1)
export const TUNINGS = {
  'Standard (EADGBE)': [40, 45, 50, 55, 59, 64],
  'Drop D (DADGBE)': [38, 45, 50, 55, 59, 64],
  'Half step down (Eb)': [39, 44, 49, 54, 58, 63],
  'Whole step down (D)': [38, 43, 48, 53, 57, 62],
  'Drop C (CGCFAD)': [36, 43, 48, 53, 57, 62],
  'Open G (DGDGBD)': [38, 43, 50, 55, 59, 62],
  'Open D (DADF#AD)': [38, 45, 50, 54, 57, 62],
  'DADGAD': [38, 45, 50, 55, 57, 62],
  '7-string (BEADGBE)': [35, 40, 45, 50, 55, 59, 64],
  'Bass 4 (EADG)': [28, 33, 38, 43],
  'Bass 5 (BEADG)': [23, 28, 33, 38, 43],
};
