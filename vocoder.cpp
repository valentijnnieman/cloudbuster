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
      adsr_attack(other.adsr_attack), adsr_decay(other.adsr_decay),
      adsr_sustain(other.adsr_sustain), adsr_release(other.adsr_release) {
  _fft_in_alloc = fft_in = (float *)fftwf_alloc_real(sig_len);
  _fft_out_alloc = fft_out = (fftwf_complex *)fftwf_alloc_complex(sig_len);
  p = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);
  pi = fftwf_plan_dft_c2r_1d(N, fft_out, fft_in, FFTW_ESTIMATE);
  PaUtil_InitializeRingBuffer(&_note_queue, sizeof(NoteEvent), 16, _note_queue_buf);
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

  _fft_in_alloc = fft_in = (float *)fftwf_alloc_real(sig_len);
  _fft_out_alloc = fft_out = (fftwf_complex *)fftwf_alloc_complex(sig_len);

  p = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);
  pi = fftwf_plan_dft_c2r_1d(N, fft_out, fft_in, FFTW_ESTIMATE);

  window = hanning(window_size, 0);

  smpl_ptr = 0;
  PaUtil_InitializeRingBuffer(&_note_queue, sizeof(NoteEvent), 16, _note_queue_buf);
}

void Vocoder::clear() {
  _adsr_phase = AdsrPhase::Attack;
  _adsr_pos = 0;
  _adsr_level = 0.0f;
  smpl_ptr = 0;
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

// get the nth sample
float Vocoder::get_sample(int note, int n) {
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
  float env;

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
      return 0.0f;
    }
    break;
  default: // Done
    running.store(false);
    return 0.0f;
  }
  if (_adsr_phase != AdsrPhase::Release)
    _adsr_level = env;

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
// Build the forward-FFT cache: one entry per analysis frame, computed once
// and shared across all notes.
// ---------------------------------------------------------------------------
void Vocoder::_build_fft_cache() {
  int hop = window_size / hop_size_div;
  int n_frames = static_cast<int>(resampled.size()) / hop + 1;
  _fft_cache.resize(n_frames);

  std::vector<float> buf(window_size, 0.0f);
  for (int f = 0; f < n_frames; f++) {
    int rp = f * hop;
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

  // Intermediate time-stretched buffer (duration × s/pr)
  size_t stretched_size = static_cast<size_t>(resampled.size() * s / pr);
  std::vector<float> stretched(stretched_size, 0.0f);

  // Synthesis window same size as analysis window — no per-frame resampling
  std::vector<float> syn_win = hanning(window_size, 0);
  float syn_sum = 0.0f;
  for (int k = 0; k * out_hop < window_size; k++)
    syn_sum += syn_win[k * out_hop] * syn_win[k * out_hop];
  if (syn_sum == 0.0f)
    syn_sum = 1.0f;

  std::vector<float> phi(N / 2 + 1, 0.0f);
  std::vector<float> prev_phase(N / 2 + 1, 0.0f);
  std::vector<std::complex<float>> fft_buf(N / 2 + 1);
  std::vector<float> ifft_buf(window_size, 0.0f);

  int wp = 0;
  for (int f = 0; f < static_cast<int>(_fft_cache.size()); f++) {
    if (wp >= static_cast<int>(stretched.size()))
      break;

    fft_buf = _fft_cache[f];
    for (int i = 0; i < N / 2 + 1; i++) {
      float phase = std::arg(fft_buf[i]);
      float amp = std::abs(fft_buf[i]);
      if (ALIEN) {
        if (i > 0) {
          float prev_amp = std::abs(fft_buf[i - 1]);
          amp = 0.9f * prev_amp + 0.1f * amp;
        }
      }
      float expected = 2.0f * M_PI * i * hop / N;
      float delta = phase - prev_phase[i] - expected;
      delta -= 2.0f * M_PI * std::round(delta / (2.0f * M_PI));

      // Phase advance: stretch s, pitch shift pr
      if (ROBOTO) {
        phi[i] += expected * s / pr;
      } else if (WHISPER) {
        phi[i] = 2.0f * M_PI * randf();
      } else {
        phi[i] += (expected + delta) * s / pr;
      }
      phi[i] -= 2.0f * M_PI * std::round(phi[i] / (2.0f * M_PI));
      prev_phase[i] = phase;
      fft_buf[i] = std::polar(amp, phi[i]);
    }

    if (ALIEN) {
      // std::vector<std::complex<float>> tmp = fft_buf;
      // for (int i = 0; i <= N / 2; i++)
      //   fft_buf[i] = tmp[N / 2 - i];
    }

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
