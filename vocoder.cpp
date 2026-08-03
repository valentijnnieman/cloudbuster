#include "vocoder.hpp"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <random>

inline float randf(float min = 0.0f, float max = 1.0f) {
  thread_local static std::mt19937 rng(std::random_device{}());
  thread_local static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return min + (max - min) * dist(rng);
}

Vocoder::Vocoder(const Vocoder &other)
    : it(other.it), N(other.N), window_size(other.window_size),
      hop_size_div(other.hop_size_div), pitch_ratio(other.pitch_ratio),
      gain(other.gain), nyquist(other.nyquist), samples(other.samples),
      resampled(other.resampled), _resampled(other._resampled),
      _buffer(other._buffer), _phi(other._phi), _prev_phase(other._prev_phase),
      _fft_buffer(other._fft_buffer), lowpass(other.lowpass),
      cutoff(other.cutoff), window(other.window),
      _syn_window(other._syn_window), _syn_window_sum(other._syn_window_sum),
      out_samples(other.out_samples), smpl_ptr(other.smpl_ptr),
      calculated(other.calculated), _calculated_until(other._calculated_until),
      sig_len(other.sig_len), analysis_hopsize(other.analysis_hopsize),
      synthesis_hopsize(other.synthesis_hopsize), fs(other.fs),
      ROBOTO(other.ROBOTO), WHISPER(other.WHISPER), ALIEN(other.ALIEN),
      current_note(other.current_note.load()), running(other.running.load()),
      stopping(other.stopping.load()), volume(other.volume.load()),
      _cancel_precompute(false), read_ptr(other.read_ptr),
      write_ptr(other.write_ptr),
      _use_precomputed(other._use_precomputed.load()),
      _precomputed(other._precomputed), _fft_cache(other._fft_cache),
      realtime(other.realtime), stretch(other.stretch),
      adsr_attack(other.adsr_attack), adsr_decay(other.adsr_decay),
      adsr_sustain(other.adsr_sustain), adsr_release(other.adsr_release) {
  // max(sig_len, N): the streaming path runs the inverse FFT straight into
  // these buffers, so they must hold a full N-point frame even when the loaded
  // file is shorter than one window.
  _fft_in_alloc = fft_in = (float *)fftwf_alloc_real(std::max(sig_len, N));
  _fft_out_alloc = fft_out =
      (fftwf_complex *)fftwf_alloc_complex(std::max(sig_len, N));
  p = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);
  pi = fftwf_plan_dft_c2r_1d(N, fft_out, fft_in, FFTW_ESTIMATE);
  PaUtil_InitializeRingBuffer(&_note_queue, sizeof(NoteEvent), 16,
                              _note_queue_buf);
  _init_streaming_buffers();
}

// Ring + scratch allocation, shared by both constructors. Runs off the audio
// thread (file load for the template, voice construction for the copies);
// after this, the streaming path never allocates.
void Vocoder::_init_streaming_buffers() {
  // Worst-case synthesis hop: stretch at its ceiling on the highest MIDI note
  // (smallest pr consumes the most stretched samples per frame).
  const float pr_min = 440.0f / note_to_freq(127);
  const int hop = window_size / hop_size_div;
  const size_t max_out_hop =
      static_cast<size_t>(static_cast<float>(hop) * STRETCH_MAX / pr_min) + 1;
  // The pump keeps the producer at most one frame past the read position, so
  // max_out_hop + window_size (+ cubic margin) live samples is the ceiling.
  size_t ring = 1;
  while (ring < max_out_hop + static_cast<size_t>(window_size) + 4)
    ring <<= 1;
  stretched.assign(ring, 0.0f);
  ring_mask = ring - 1;

  _fft_scratch.resize(static_cast<size_t>(N / 2 + 1));
  _ifft_scratch.assign(static_cast<size_t>(N), 0.0f);
  _prev_frame.assign(static_cast<size_t>(N / 2 + 1), {0.0f, 0.0f});
}

void Vocoder::print_stats() {
  std::cout << "samples: " << samples.size();
  std::cout << std::endl;
  // std::cout << "window[0]: " << window[0] << std::endl;
}
Vocoder::Vocoder(const std::string &filename, int N, int window_size,
                 int hop_size_div, float samplerate)
    : it(it), N(N), window_size(window_size), hop_size_div(hop_size_div),
      calculated(false), lowpass(BiquadLowpass(samplerate, N)) {
  /* Open the soundfile */
  file = sf_open(filename.c_str(), SFM_READ, &info);
  samples.resize(info.frames * info.channels);
  sf_readf_float(file, &samples[0], info.frames);

  // Mix down to mono regardless of source channel count
  std::vector<float> mono(info.frames);
  for (sf_count_t i = 0; i < info.frames; i++) {
    float sum = 0.0f;
    for (int c = 0; c < info.channels; c++)
      sum += samples[i * info.channels + c];
    mono[i] = sum / info.channels;
  }
  samples.clear();

  fs = samplerate;
  double src_ratio = fs / info.samplerate;
  int out_frames = static_cast<int>(info.frames * src_ratio);
  resampled.resize(out_frames, 0.0f);

  int mono_sz = static_cast<int>(mono.size());
  for (int i = 0; i < out_frames; i++) {
    float pos = i / src_ratio;
    int idx = static_cast<int>(pos);
    float frac = pos - idx;
    float y0 = (idx > 0) ? mono[idx - 1] : 0.0f;
    float y1 = (idx < mono_sz) ? mono[idx] : 0.0f;
    float y2 = (idx + 1 < mono_sz) ? mono[idx + 1] : 0.0f;
    float y3 = (idx + 2 < mono_sz) ? mono[idx + 2] : 0.0f;
    resampled[i] = cubic_interp(y0, y1, y2, y3, frac);
  }

  /* Close the soundfile */
  sf_close(file);

  pitch_ratio = 1.0f;
  out_samples = std::vector<float>((resampled.size()), 0.0f);
  _buffer = std::vector<float>(window_size, 0.0f);
  _phi = std::vector<float>(N / 2 + 1, 0.0f);
  _prev_phase = std::vector<float>(N / 2 + 1, 0.0f);
  _fft_buffer = std::vector<std::complex<float>>(N / 2 + 1, 0.0f);

  nyquist = fs / 2.0f;
  cutoff = nyquist / 2.0f;

  sig_len = resampled.size();

  // max(sig_len, N): the streaming path runs the inverse FFT straight into
  // these buffers, so they must hold a full N-point frame even when the loaded
  // file is shorter than one window.
  _fft_in_alloc = fft_in = (float *)fftwf_alloc_real(std::max(sig_len, N));
  _fft_out_alloc = fft_out =
      (fftwf_complex *)fftwf_alloc_complex(std::max(sig_len, N));

  p = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);
  pi = fftwf_plan_dft_c2r_1d(N, fft_out, fft_in, FFTW_ESTIMATE);

  window = hanning(window_size, 0);

  smpl_ptr = 0;
  PaUtil_InitializeRingBuffer(&_note_queue, sizeof(NoteEvent), 16,
                              _note_queue_buf);

  _init_streaming_buffers();
}

void Vocoder::clear() {
  _adsr_phase = AdsrPhase::Attack;
  _adsr_pos = 0;
  _adsr_level = 0.0f;
  smpl_ptr = 0;

  // Streaming state — reset per note (clear() runs in the note-on handling on
  // the audio thread, so nothing here may allocate).
  std::fill(_phi.begin(), _phi.end(), 0.0f);
  std::fill(_prev_frame.begin(), _prev_frame.end(),
            std::complex<float>(0.0f, 0.0f));
  frame_idx = 0;
  wp = 0;
  ready = 0;
  written_end = 0;
  read_pos = 0.0;
  first_frame = true;
  // Belt and braces: written_end already forces every cell to be zeroed before
  // its first OLA write, but a clean ring keeps stale audio out of any future
  // debugging. std::fill, not reassignment — no allocation.
  std::fill(stretched.begin(), stretched.end(), 0.0f);
}

std::vector<float> Vocoder::hanning(int N, short itype) {
  int half, i, idx, n;
  std::vector<float> w(N, 0.0f);

  // w = (float*) calloc(N, sizeof(float));
  // memset(w, 0, N*sizeof(float));

  if (itype == 1) // periodic function
    n = N - 1;
  else
    n = N;

  if (n % 2 == 0) {
    half = n / 2;
    for (i = 0; i < half;
         i++) // CALC_HANNING   Calculates Hanning window samples.
      w[i] = 0.5 * (1 - cos(2 * M_PI * (i + 1) / (n + 1)));

    idx = half - 1;
    for (i = half; i < n; i++) {
      w[i] = w[idx];
      idx--;
    }
  } else {
    half = (n + 1) / 2;
    for (i = 0; i < half;
         i++) // CALC_HANNING   Calculates Hanning window samples.
      w[i] = 0.5 * (1 - cos(2 * M_PI * (i + 1) / (n + 1)));

    idx = half - 2;
    for (i = half; i < n; i++) {
      w[i] = w[idx];
      idx--;
    }
  }

  if (itype == 1) // periodic function
  {
    for (i = N - 1; i >= 1; i--)
      w[i] = w[i - 1];
    w[0] = 0.0;
  }
  return (w);
}

float Vocoder::note_to_freq(int note) {
  float a = 440.0f; // frequency of A (conmon value is 440Hz)
  // float d = 587.33; // freq of D note
  // return a * 2^((note−69)/12);

  return a * pow(2.0f, ((note - 69.0f) / 12.0f));
}

void Vocoder::forward_fft(float *time_data, std::complex<float> *freq_data) {
  fft_out = reinterpret_cast<fftwf_complex *>(freq_data);
  fft_in = time_data;
  fftwf_execute_dft_r2c(p, fft_in, fft_out);
}

void Vocoder::ifft(float *time_data, std::complex<float> *freq_data) {
  fft_out = reinterpret_cast<fftwf_complex *>(freq_data);
  fft_in = time_data;
  fftwf_execute_dft_c2r(pi, fft_out, fft_in);
}

std::vector<float> Vocoder::lowpass_filter(std::vector<float> input,
                                           float cutoff) {
  std::vector<float> output(input.size());
  float RC = 1.0f / (2.0f * M_PI * cutoff);
  float dt = 1.0f / fs;
  float alpha = dt / (RC + dt);

  output[0] = input[0];
  for (size_t i = 1; i < input.size(); ++i) {
    output[i] = output[i - 1] + alpha * (input[i] - output[i - 1]);
  }
  return output;
}

// Cubic interpolation for resampling
float Vocoder::cubic_interp(float y0, float y1, float y2, float y3, float mu) {
  float a0 = y3 - y2 - y0 + y1;
  float a1 = y0 - y1 - a0;
  float a2 = y2 - y0;
  float a3 = y1;
  return a0 * mu * mu * mu + a1 * mu * mu + a2 * mu + a3;
}

// Advance the ADSR one sample. Shared by both synthesis paths (get_sample and
// stream_sample); returns false when the voice has finished sounding.
bool Vocoder::_adsr_env(float &env) {
  // Transition to Release on note-off
  if (stopping.load() && _adsr_phase != AdsrPhase::Release &&
      _adsr_phase != AdsrPhase::Done) {
    _adsr_phase = AdsrPhase::Release;
    _adsr_pos = 0;
  }

  // Compute ADSR envelope for this sample
  int atk = std::max(1, static_cast<int>(adsr_attack * fs));
  int dec = std::max(1, static_cast<int>(adsr_decay * fs));
  int rel = std::max(1, static_cast<int>(adsr_release * fs));

  switch (_adsr_phase) {
  case AdsrPhase::Attack:
    env = static_cast<float>(_adsr_pos) / atk;
    if (++_adsr_pos >= atk) {
      _adsr_phase = AdsrPhase::Decay;
      _adsr_pos = 0;
    }
    break;
  case AdsrPhase::Decay:
    env = 1.0f - (1.0f - adsr_sustain) * static_cast<float>(_adsr_pos) / dec;
    if (++_adsr_pos >= dec) {
      _adsr_phase = AdsrPhase::Sustain;
      _adsr_pos = 0;
    }
    break;
  case AdsrPhase::Sustain:
    env = adsr_sustain;
    break;
  case AdsrPhase::Release:
    env = _adsr_level * (1.0f - static_cast<float>(_adsr_pos) / rel);
    if (++_adsr_pos >= rel) {
      running.store(false);
      stopping.store(false);
      current_note.store(0);
      smpl_ptr = 0;
      _adsr_phase = AdsrPhase::Done;
      return false;
    }
    break;
  default: // Done
    running.store(false);
    return false;
  }
  if (_adsr_phase != AdsrPhase::Release)
    _adsr_level = env;

  return true;
}

// get the nth sample (precomputed path)
float Vocoder::get_sample(int note, int n) {
  float env;
  if (!_adsr_env(env))
    return 0.0f;

  if (_use_precomputed) {
    auto it = _precomputed.find(note);
    if (it != _precomputed.end()) {
      if (n >= static_cast<int>(it->second.size())) {
        running.store(false);
        stopping.store(false);
        current_note.store(0);
        smpl_ptr = 0;
        return 0.0f;
      }
      return it->second[n] * volume.load() * env;
    }
  }

  // Precomputed data not yet available — silent until ready
  return 0.0f;
}

// ---------------------------------------------------------------------------
// Real-time path: render the next output sample for `note`, synthesizing
// frames out of _fft_cache on demand instead of reading a baked buffer.
// ---------------------------------------------------------------------------
float Vocoder::stream_sample(int note) {
  float env;
  if (!_adsr_env(env))
    return 0.0f;

  if (_fft_cache.empty())
    return 0.0f;

  float pr = 440.0f / note_to_freq(note);

  // Pump: cubic_interp reads stretched[idx-1 .. idx+2], so production must
  // stay at least 3 samples ahead of the integer read position. `needed` is
  // in stretched-domain samples, not output samples — the two advance at
  // different rates. The ring is sized so this lookahead (one frame past the
  // read position at most) always fits without overwriting unread samples.
  const int64_t idx = static_cast<int64_t>(read_pos);
  const int64_t needed = idx + 3;
  while (running.load() && ready < needed)
    _synth_one_frame(note, pi, _fft_in_alloc, _fft_out_alloc);

  // Walk exhausted before reaching the read position: nothing left to play.
  if (ready < needed)
    return 0.0f;

  // Streaming pitch resample: consume the stretched signal at 1/pr samples per
  // output sample — the streaming equivalent of the batch path's final
  // resample loop, which reads at pos = i / pr. Reading faster than 1x raises
  // pitch, slower lowers it; combined with the out_hop time-stretch in
  // _synth_one_frame, duration stays x stretch per note.
  const size_t m = ring_mask;
  float mu = static_cast<float>(read_pos - static_cast<double>(idx));
  float y0 = (idx > 0) ? stretched[static_cast<size_t>(idx - 1) & m] : 0.0f;
  float y1 = stretched[static_cast<size_t>(idx) & m];
  float y2 = stretched[static_cast<size_t>(idx + 1) & m];
  float y3 = stretched[static_cast<size_t>(idx + 2) & m];
  float sample = cubic_interp(y0, y1, y2, y3, mu);
  // `stretched` and the output stream are both at fs (the file is resampled to
  // fs on load and PortAudio runs at fs), so pitch shifting is the only rate
  // conversion here.
  read_pos += 1.0 / pr;

  return sample * volume.load() * env;
}

// ---------------------------------------------------------------------------
// Build the forward-FFT cache: one entry per analysis frame, computed once
// and shared across all notes.
// ---------------------------------------------------------------------------
void Vocoder::_build_fft_cache() {
  int hop = window_size / hop_size_div;
  int n_frames = static_cast<int>(resampled.size()) / hop + 1;
  _fft_cache.resize(n_frames);

  std::vector<float> buf(N, 0.0f); // N-sized so the FFTW plan reads valid
                                   // memory when window_size < N
  for (int f = 0; f < n_frames; f++) {
    int rp = f * hop;
    std::fill(buf.begin(), buf.end(), 0.0f); // clear zero-padding each frame
    for (int i = 0; i < window_size; i++) {
      int idx = rp + i;
      buf[i] = (idx < static_cast<int>(resampled.size()))
                   ? window[i] * resampled[idx]
                   : 0.0f;
    }
    _fft_cache[f].resize(N / 2 + 1);
    forward_fft(buf.data(), _fft_cache[f].data());
  }
}

// Analysis pass for the real-time path. Run it on the template vocoder before
// the voices are copied off it — the copy constructor carries _fft_cache along,
// so the analysis is paid once per file instead of once per voice.
void Vocoder::prepare_realtime() { _build_fft_cache(); }

// ---------------------------------------------------------------------------
// Synthesize one frame of the walk through _fft_cache into the ring. Runs on
// the audio thread at hop rate — no allocation, no locks. This is the
// streaming counterpart of the per-frame loop body in _synth_note, with two
// differences: phase advance is peak-locked (Laroche-Dolson) rather than
// per-bin, and the OLA lands in a fixed ring instead of a whole-file buffer.
// ---------------------------------------------------------------------------
void Vocoder::_synth_one_frame(int note, fftwf_plan inv, float *fi,
                               fftwf_complex *fo) {
  float freq = note_to_freq(note);
  float pr = 440.0f / freq;

  int hop = window_size / hop_size_div;
  // Synthesis hop scaled by 1/pr → time-stretch factor 1/pr. The streaming
  // resample in stream_sample then restores duration while shifting pitch.
  // Clamped to STRETCH_MAX: the ring was sized for that worst case, and -s
  // takes an unbounded value.
  float s = std::min(stretch, STRETCH_MAX);
  int out_hop = std::max(1, static_cast<int>(hop * s / pr));
  // Phase advance must track the *realized* (integer) hop ratio, not the
  // unquantized target s/pr, or the synthesis phase drifts out of sync with
  // where frames are actually placed — audible as a small per-note mistuning.
  float step_ratio = static_cast<float>(out_hop) / static_cast<float>(hop);

  // Synthesis window: identical to the analysis window (same hanning call), so
  // reuse the member instead of rebuilding it every frame.
  const std::vector<float> &syn_win = window;

  // Steady-state overlap-add normalisation: divide by the *peak* of the
  // syn_win^2 overlap envelope over one hop period — the streaming replacement
  // for the batch path's syn_sum, which edge-samples that envelope. With full
  // overlap the envelope is flat, so this equals the usual COLA constant; when
  // out_hop grows past window_size and frames stop overlapping, it pins frame
  // centres at source level instead of blowing up. The envelope has period
  // out_hop, so scanning one period costs O(window).
  float cola = 0.0f;
  const int period = std::min(out_hop, window_size);
  for (int d = 0; d < period; d++) {
    float e = 0.0f;
    for (int k = d; k < window_size; k += out_hop)
      e += syn_win[k] * syn_win[k];
    cola = std::max(cola, e);
  }
  cola = std::max(cola, 1e-6f);

  // Member scratch, not locals: this runs at hop rate on the audio thread.
  std::vector<std::complex<float>> &fft_buf = _fft_scratch;
  std::vector<float> &ifft_buf = _ifft_scratch;

  const int n_frames = static_cast<int>(_fft_cache.size());
  fft_buf = _fft_cache[std::clamp(frame_idx, 0, n_frames - 1)];

  if (ALIEN) {
    // Recursive magnitude smear up the spectrum, as in _synth_note: each bin
    // takes 90% of its (already smeared) neighbour's magnitude. Scaling the
    // bin leaves its phase untouched, so this stays a magnitude-only effect.
    for (int i = 1; i < N / 2 + 1; i++) {
      float prev_amp = std::abs(fft_buf[i - 1]);
      float amp = std::abs(fft_buf[i]);
      float smeared = 0.9f * prev_amp + 0.1f * amp;
      fft_buf[i] = (amp > 1e-20f) ? fft_buf[i] * (smeared / amp)
                                  : std::complex<float>(smeared, 0.0f);
    }
  }

  find_peaks(fft_buf, _peaks_scratch);
  get_region_boundaries(_peaks_scratch, N / 2 + 1, _bounds_scratch);
  shift_peaks(fft_buf, _peaks_scratch, _bounds_scratch, _phi, _prev_frame, hop,
              step_ratio, first_frame);
  first_frame = false;

  // IFFT — fftwf_execute_dft_c2r is thread-safe with private buffers
  for (int i = 0; i < N / 2 + 1; i++) {
    fo[i][0] = fft_buf[i].real();
    fo[i][1] = fft_buf[i].imag();
  }
  fftwf_execute_dft_c2r(inv, fo, fi);
  for (int i = 0; i < N; i++)
    ifft_buf[i] = fi[i];

  // Zero the ring cells this frame reaches that haven't been reset since they
  // were last read — the streaming stand-in for "a fresh buffer is zero", and
  // it handles wraparound reuse. Starting from written_end (not wp) also clears
  // any gap when out_hop > window_size.
  const int64_t frame_end = wp + window_size;
  for (int64_t p = written_end; p < frame_end; p++)
    stretched[static_cast<size_t>(p) & ring_mask] = 0.0f;
  if (frame_end > written_end)
    written_end = frame_end;

  // OLA into the ring, pre-scaled by the COLA constant
  for (int i = 0; i < window_size; i++)
    stretched[static_cast<size_t>(wp + i) & ring_mask] +=
        syn_win[i] * ifft_buf[i] / (N * cola);

  wp += out_hop;
  ready = wp;

  frame_idx++;
  if (frame_idx >= n_frames)
    running.store(false); // walked off the end of the file
}

// ---------------------------------------------------------------------------
// Finds peaks in FFT frame, as per Laroche-Dolson paper
// (https://www.ee.columbia.edu/~dpwe/papers/LaroD99-pvoc.pdf)
// ---------------------------------------------------------------------------
void Vocoder::find_peaks(std::vector<std::complex<float>> &bins,
                         std::vector<int> &peaks) {
  peaks.clear(); // keeps capacity — called per frame on the audio thread

  // Squared magnitudes (std::norm) throughout: x > y iff x^2 > y^2 for
  // non-negatives, so the peak comparisons are unchanged but the sqrt inside
  // std::abs disappears — this runs over every bin at frame rate.
  float max_mag = 0.0f;
  for (int i = 2; i < (int)bins.size() - 2; i++) {
    float mag = std::norm(bins[i]);
    if (mag > max_mag)
      max_mag = mag;
  }

  // |b| > 0.01 * max  <=>  |b|^2 > 1e-4 * max^2
  float treshold_mag = max_mag * 0.0001f;
  for (int i = 2; i < (int)bins.size() - 2; i++) {
    float mag = std::norm(bins[i]);
    if (mag > std::norm(bins[i - 1]) && mag > std::norm(bins[i + 1]) &&
        mag > std::norm(bins[i - 2]) && mag > std::norm(bins[i + 2]) &&
        mag > treshold_mag) {
      peaks.push_back(i);
    }
  }
}

void Vocoder::get_region_boundaries(const std::vector<int> &peaks, int M,
                                    std::vector<int> &boundaries) {
  // Fills boundary[i] = first bin belonging to peak i
  // so peak i owns bins [boundary[i], boundary[i+1])
  boundaries.clear(); // keeps capacity — called per frame on the audio thread
  boundaries.push_back(0);

  for (int i = 0; i + 1 < (int)peaks.size(); i++) {
    int mid = (peaks[i] + peaks[i + 1]) / 2;
    boundaries.push_back(mid);
  }
  boundaries.push_back(M);
}

// Laroche-Dolson phase locking.
// phi[k]       – accumulated synthesis phase per peak bin.
// prev_frame   – previous *analysis* frame, complex (updated here for next
//                call); its phase is only needed at the current peaks, so
//                arg() is computed lazily instead of atan2-ing every bin.
// step         – synthesis-to-analysis hop ratio = out_hop / hop.
void Vocoder::shift_peaks(std::vector<std::complex<float>> &frame,
                          const std::vector<int> &peaks,
                          const std::vector<int> &boundaries,
                          std::vector<float> &phi,
                          std::vector<std::complex<float>> &prev_frame, int hop,
                          float step, bool first_frame) {
  if (first_frame) {
    // Seed phi directly from analysis phases so the first synthesis frame
    // exactly matches the analysis frame, avoiding a phase discontinuity at
    // the frame-0 / frame-1 boundary.
    for (int k : peaks)
      phi[k] = std::arg(frame[k]);
  } else {
    // IF tracking: advance synthesis phase for each peak
    for (int k : peaks) {
      float phase = std::arg(frame[k]);
      float expected = 2.0f * M_PI * k * hop / N;
      float delta = phase - std::arg(prev_frame[k]) - expected;
      delta -= 2.0f * M_PI * std::round(delta / (2.0f * M_PI));

      if (ROBOTO) {
        phi[k] += expected * step;
      } else if (WHISPER) {
        phi[k] = 2.0f * M_PI * randf();
      } else {
        phi[k] += (expected + delta) * step;
      }

      phi[k] -= 2.0f * M_PI * std::round(phi[k] / (2.0f * M_PI));
    }
  }

  // Record the analysis frame before modifying it (next call's prev_frame).
  // Vector assignment reuses capacity — no allocation.
  prev_frame = frame;

  // Phase locking (Laroche-Dolson): rotate each bin by the same amount as its
  // region's peak, preserving inter-bin relative phases. The rotation
  // polar(|b|, phi[k] + arg(b) - arg(k)) == b * e^{i(phi[k] - arg(k))} costs
  // one polar per *region* and a complex multiply per bin — no per-bin
  // atan2/abs/polar. arg(frame[k]) is still the unrotated analysis phase here
  // because each region is only rotated after its own rot is computed.
  for (int p = 0; p < (int)peaks.size(); p++) {
    int k = peaks[p];
    const std::complex<float> rot =
        std::polar(1.0f, phi[k] - std::arg(frame[k]));
    for (int b = boundaries[p]; b < boundaries[p + 1]; b++)
      frame[b] *= rot;
  }
}

// ---------------------------------------------------------------------------
// Per-note synthesis using the shared FFT cache.
// inv/fi/fo are per-thread FFTW resources created before thread launch.
// ---------------------------------------------------------------------------
std::vector<float> Vocoder::_synth_note(int note, fftwf_plan inv, float *fi,
                                        fftwf_complex *fo) {
  float freq = note_to_freq(note);
  float pr = 440.0f / freq;

  int hop = window_size / hop_size_div;
  // Synthesis hop scaled by 1/pr → time-stretch factor 1/pr.
  // A final resample by pr then restores original duration while shifting
  // pitch.
  float s = stretch;
  int out_hop = std::max(1, static_cast<int>(hop * s / pr));
  // Phase advance must track the *realized* (integer) hop ratio, not the
  // unquantized target s/pr, or the synthesis phase drifts out of sync with
  // where frames are actually placed — audible as a small per-note mistuning.
  float step_ratio = static_cast<float>(out_hop) / static_cast<float>(hop);

  // Intermediate time-stretched buffer (duration × s/pr)
  size_t stretched_size = static_cast<size_t>(resampled.size() * s / pr);
  std::vector<float> stretched(stretched_size, 0.0f);

  // Synthesis window same size as analysis window — no per-frame resampling
  std::vector<float> syn_win = hanning(window_size, 0);
  // Overlap-add normalisation: the *peak* of the syn_win^2 overlap envelope
  // over one hop period (same constant the streaming path computes). Sampling
  // that envelope only at offset 0, as this used to, collapses to syn_win[0]^2
  // (~1e-10 for a hanning window) once out_hop >= window_size and frames stop
  // overlapping — a ~1e9 gain blow-up on high notes at large stretch. Taking
  // the peak pins frame centres at source level instead; with normal overlap
  // the envelope is flat, so the value is unchanged.
  float syn_sum = 0.0f;
  const int period = std::min(out_hop, window_size);
  for (int d = 0; d < period; d++) {
    float e = 0.0f;
    for (int k = d; k < window_size; k += out_hop)
      e += syn_win[k] * syn_win[k];
    syn_sum = std::max(syn_sum, e);
  }
  syn_sum = std::max(syn_sum, 1e-6f);

  std::vector<float> phi(N / 2 + 1, 0.0f);
  std::vector<std::complex<float>> fft_buf(N / 2 + 1);
  std::vector<float> ifft_buf(window_size, 0.0f);
  // Locals, not the _peaks_scratch/_prev_frame members: precompute runs this
  // on several worker threads at once against the same Vocoder.
  std::vector<int> peaks, boundaries;
  std::vector<std::complex<float>> prev_frame(N / 2 + 1, {0.0f, 0.0f});
  bool first = true;

  int wp = 0;
  for (int f = 0; f < static_cast<int>(_fft_cache.size()); f++) {
    if (wp >= static_cast<int>(stretched.size()))
      break;

    fft_buf = _fft_cache[f];

    if (ALIEN) {
      // Recursive magnitude smear up the spectrum: each bin takes 90% of its
      // (already smeared) neighbour's magnitude. Hoisted out of the phase loop
      // so the phase-locked path below sees the smeared magnitudes too.
      for (int i = 1; i < N / 2 + 1; i++) {
        float prev_amp = std::abs(fft_buf[i - 1]);
        float amp = std::abs(fft_buf[i]);
        float smeared = 0.9f * prev_amp + 0.1f * amp;
        fft_buf[i] = (amp > 1e-20f) ? fft_buf[i] * (smeared / amp)
                                    : std::complex<float>(smeared, 0.0f);
      }
    }

    // Laroche-Dolson peak locking, the same call the streaming path makes
    // (ROBOTO/WHISPER included — shift_peaks branches on them internally).
    // Without locking, every bin propagates its own phase and the partials of
    // one note drift apart: the smeared, chorused "two of everything" sound.
    find_peaks(fft_buf, peaks);
    get_region_boundaries(peaks, N / 2 + 1, boundaries);
    shift_peaks(fft_buf, peaks, boundaries, phi, prev_frame, hop, step_ratio,
                first);
    first = false;

    // IFFT — fftwf_execute_dft_c2r is thread-safe with private buffers
    for (int i = 0; i < N / 2 + 1; i++) {
      fo[i][0] = fft_buf[i].real();
      fo[i][1] = fft_buf[i].imag();
    }
    fftwf_execute_dft_c2r(inv, fo, fi);
    for (int i = 0; i < window_size; i++)
      ifft_buf[i] = fi[i];

    // OLA into time-stretched buffer (no per-frame resampling)
    for (int i = 0;
         i < window_size && (wp + i) < static_cast<int>(stretched.size()); i++)
      stretched[wp + i] += (syn_win[i] * ifft_buf[i]) / (syn_sum * N);

    wp += out_hop;
  }

  // Final resample by pr: pitch-shifts while keeping duration × s
  size_t out_size = static_cast<size_t>(resampled.size() * s);
  std::vector<float> out(out_size, 0.0f);
  for (size_t i = 0; i < out_size; i++) {
    float pos = i / pr;
    int idx = static_cast<int>(pos);
    float frac = pos - idx;
    float y0 = (idx > 0 && idx - 1 < (int)stretched_size) ? stretched[idx - 1] : 0.0f;
    float y1 = (idx < (int)stretched_size) ? stretched[idx] : 0.0f;
    float y2 = (idx + 1 < (int)stretched_size) ? stretched[idx + 1] : 0.0f;
    float y3 = (idx + 2 < (int)stretched_size) ? stretched[idx + 2] : 0.0f;
    out[i] = cubic_interp(y0, y1, y2, y3, frac);
  }

  return out;
}

void Vocoder::apply_precomputed_from(const Vocoder &source) {
  _precomputed = source._precomputed;
  _use_precomputed.store(true);
}

void Vocoder::precompute(int min_note, int max_note) {
  _cancel_precompute.store(false);
  int n_notes = max_note - min_note + 1;
  int n_threads = std::min(
      std::max(1, static_cast<int>(std::thread::hardware_concurrency())),
      n_notes);

  _build_fft_cache();

  // Create per-thread FFTW resources BEFORE spawning threads —
  // fftwf_plan_dft_c2r_1d is NOT thread-safe.
  struct FftwCtx {
    float *fi;
    fftwf_complex *fo;
    fftwf_plan inv;
  };
  std::vector<FftwCtx> ctx(n_threads);
  for (int t = 0; t < n_threads; t++) {
    ctx[t].fi = (float *)fftwf_alloc_real(N);
    ctx[t].fo = (fftwf_complex *)fftwf_alloc_complex(N);
    ctx[t].inv = fftwf_plan_dft_c2r_1d(N, ctx[t].fo, ctx[t].fi, FFTW_ESTIMATE);
  }

  std::vector<std::map<int, std::vector<float>>> results(n_threads);
  std::vector<std::thread> threads;
  std::mutex log_mtx;

  for (int t = 0; t < n_threads; t++) {
    int t_min = min_note + t * n_notes / n_threads;
    int t_max = (t == n_threads - 1)
                    ? max_note
                    : min_note + (t + 1) * n_notes / n_threads - 1;

    threads.emplace_back([this, &ctx, &results, &log_mtx, t, t_min, t_max]() {
      for (int note = t_min; note <= t_max; note++) {
        if (_cancel_precompute.load())
          return;
        results[t][note] = _synth_note(note, ctx[t].inv, ctx[t].fi, ctx[t].fo);
        std::lock_guard<std::mutex> lock(log_mtx);
      }
    });
  }

  for (auto &th : threads)
    th.join();

  for (int t = 0; t < n_threads; t++) {
    fftwf_destroy_plan(ctx[t].inv);
    fftwf_free(ctx[t].fi);
    fftwf_free(ctx[t].fo);
  }

  _fft_cache.clear();

  if (_cancel_precompute.load()) {
    return;
  }

  for (auto &r : results)
    for (auto &[note, buf] : r)
      _precomputed[note] = std::move(buf);

  _use_precomputed.store(true);
  // std::cout << "precompute done" << std::endl;
}

Vocoder::~Vocoder() {
  fftwf_destroy_plan(p);
  fftwf_destroy_plan(pi);

  fftwf_free(_fft_in_alloc);
  fftwf_free(_fft_out_alloc);
}
