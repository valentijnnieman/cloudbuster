#include "vocoder.hpp"
#include <mutex>

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
      PHI_UNWRAP(other.PHI_UNWRAP), GONGO(other.GONGO),
      INSTANTANEOUS(other.INSTANTANEOUS),
      current_note(other.current_note.load()),
      pending_note(other.pending_note.load()),
      clear_pending(other.clear_pending.load()), running(other.running.load()),
      stopping(other.stopping.load()), read_ptr(other.read_ptr),
      write_ptr(other.write_ptr), _use_precomputed(other._use_precomputed),
      _precomputed(other._precomputed), _fft_cache(other._fft_cache) {
  fft_in = (float *)fftwf_alloc_real(sig_len);
  fft_out = (fftwf_complex *)fftwf_alloc_complex(sig_len);
  p = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);
  pi = fftwf_plan_dft_c2r_1d(N, fft_out, fft_in, FFTW_ESTIMATE);
}

void Vocoder::print_stats() {
  std::cout << "samples size: " << samples.size();
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

  SRC_DATA src_data;
  src_data.data_in = mono.data();
  src_data.data_out = resampled.data();
  src_data.input_frames = info.frames;
  src_data.output_frames = out_frames;
  src_data.src_ratio = src_ratio;
  src_data.end_of_input = 1;
  src_simple(&src_data, SRC_SINC_FASTEST, 1);

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

  fft_in = (float *)fftwf_alloc_real(sig_len);
  fft_out = (fftwf_complex *)fftwf_alloc_complex(sig_len);

  p = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);
  pi = fftwf_plan_dft_c2r_1d(N, fft_out, fft_in, FFTW_ESTIMATE);

  window = hanning(window_size, 0);

  smpl_ptr = 0;
}

void Vocoder::clear() {
  if (_use_precomputed && _precomputed.count(current_note.load())) {
    smpl_ptr = 0;
    return;
  }

  /*out_samples.clear();*/

  float freq = note_to_freq(current_note);
  float div = 440.0f;
  float r = div / freq;
  pitch_ratio = r;

  gain = 1.0f;

  out_samples = std::vector<float>((resampled.size() * pitch_ratio), 0.0f);
  /*_buffer = std::vector<float>(window_size, 0.0f);*/
  std::fill(_phi.begin(), _phi.end(), 0.0f);
  std::fill(_prev_phase.begin(), _prev_phase.end(), 0.0f);

  size_t syn_size = static_cast<size_t>(window_size * pitch_ratio);
  _syn_window = hanning(static_cast<int>(syn_size), 0);

  // Correct OLA normalization: sum of squared window values at output hop
  // positions. For Hanning with hop_size_div× overlap this equals 0.375 *
  // hop_size_div (≈3 for div=8), NOT the simple window sum (≈N/2), which would
  // be ~170× too large.
  int out_h =
      std::max(1, static_cast<int>((window_size / hop_size_div) * pitch_ratio));
  _syn_window_sum = 0.0f;
  for (int k = 0; k * out_h < static_cast<int>(_syn_window.size()); k++)
    _syn_window_sum += _syn_window[k * out_h] * _syn_window[k * out_h];
  if (_syn_window_sum == 0.0f)
    _syn_window_sum = 1.0f; // guard against empty window

  // Clamp cutoff below Nyquist: only anti-alias when compressing (pitch_ratio <
  // 1)
  lowpass.setCutoff(std::min(nyquist * pitch_ratio, nyquist * 0.99f));
  lowpass.reset();

  calculated = false;
  _calculated_until = 0;
  read_ptr = 0;
  write_ptr = 0;
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

void Vocoder::resample(float *data_in, float *data_out, int input_size,
                       int output_size, float pitch_ratio) {

  SRC_DATA src_data;
  src_data.data_in = data_in;
  src_data.data_out = data_out;
  src_data.input_frames = input_size;
  src_data.output_frames = output_size;
  src_data.src_ratio = pitch_ratio;

  src_simple(&src_data, 2, 2);
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
      return it->second[n];
    }
  }

  if (n >= _calculated_until) {
    // Prepare new window
    size_t resampled_size = static_cast<size_t>(window_size * pitch_ratio);
    if (_resampled.size() != resampled_size)
      _resampled.resize(resampled_size, 0.0f);

    synthesis_hopsize = window_size / hop_size_div;
    analysis_hopsize =
        synthesis_hopsize; // equal hops; pitch shift via per-window resampling

    // Windowed samples
    for (size_t i = 0; i < window_size; i++) {
      size_t sample_idx = read_ptr + i;
      _buffer[i] =
          (sample_idx < resampled.size() ? window[i] * resampled[sample_idx]
                                         : 0.0f);
    }

    /*auto filtered_buffer = lowpass_filter(_buffer, (_fft_buffer.size() *
     * pitch_ratio) / 2);*/
    // FFT
    forward_fft(_buffer.data(),
                reinterpret_cast<std::complex<float> *>(_fft_buffer.data()));

    // Phase vocoder processing
    for (size_t i = 0; i < _fft_buffer.size(); i++) {
      float phase = std::arg(_fft_buffer[i]);
      float amplitude = std::abs(_fft_buffer[i]);

      // Expected phase advance for this bin over one analysis hop
      float expected =
          2.0f * M_PI * static_cast<float>(i) * analysis_hopsize / N;

      // True phase deviation, wrapped to [-pi, pi]
      float delta = phase - _prev_phase[i] - expected;
      delta -= 2.0f * M_PI * std::round(delta / (2.0f * M_PI));

      // Accumulate output phase and store current phase for next frame
      _phi[i] += expected + delta;
      _phi[i] -= 2.0f * M_PI * std::round(_phi[i] / (2.0f * M_PI));
      _prev_phase[i] = phase;

      _fft_buffer[i].real(amplitude * std::cos(_phi[i]));
      _fft_buffer[i].imag(amplitude * std::sin(_phi[i]));
    }

    // Inverse FFT
    ifft(_buffer.data(),
         reinterpret_cast<std::complex<float> *>(_fft_buffer.data()));

    // Resample
    size_t s = _buffer.size() - 1;
    size_t L = static_cast<size_t>(std::floor(s * pitch_ratio));
    for (size_t i = 0; i < L; i++) {
      /*float x = static_cast<float>(i) * s / L;*/
      float x = static_cast<float>(i) * (s - 1) / (L - 1);
      size_t ix = static_cast<size_t>(std::floor(x));
      float mu = x - ix;
      // Clamp indices to valid range
      size_t buf_size = _buffer.size();
      size_t ix0 = std::min((ix == 0 ? 0 : ix - 1), buf_size - 1);
      size_t ix1 = std::min(ix, buf_size - 1);
      size_t ix2 = std::min((ix + 1 < s ? ix + 1 : s - 1), buf_size - 1);
      size_t ix3 = std::min((ix + 2 < s ? ix + 2 : s - 1), buf_size - 1);

      _resampled[i] = lowpass.process(cubic_interp(
          _buffer[ix0], _buffer[ix1], _buffer[ix2], _buffer[ix3], mu));
    }

    // low pass filter
    /*auto filtered_buffer = lowpass_filter(_resampled, fs / 4);*/

    // Overlap-add into output domain (write_ptr advances at synthesis *
    // pitch_ratio rate)
    for (size_t i = 0;
         i < _resampled.size() && (write_ptr + i) < out_samples.size(); i++) {
      out_samples[write_ptr + i] +=
          (_syn_window[i] * _resampled[i]) / (_syn_window_sum * N);
    }

    // Increment pointers (out_hop >= 1 to prevent infinite loop)
    int out_hop =
        std::max(1, static_cast<int>(synthesis_hopsize * pitch_ratio));
    read_ptr += synthesis_hopsize;
    write_ptr += out_hop;
    _calculated_until += out_hop;

    if (write_ptr >= static_cast<int>(out_samples.size())) {
      calculated = true;
      _calculated_until = static_cast<int>(out_samples.size());
    }
  }

  if (n >= out_samples.size()) {
    running.store(false);
    stopping.store(false);

    current_note.store(0);
    smpl_ptr = 0;
    read_ptr = 0;
    write_ptr = 0;

    clear();
    return 0.0f;
  }

  return out_samples[n] / gain;
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

  size_t out_size = static_cast<size_t>(resampled.size() * pr);
  std::vector<float> out(out_size, 0.0f);

  int hop = window_size / hop_size_div;
  int out_hop = std::max(1, static_cast<int>(hop * pr));

  size_t syn_size = static_cast<size_t>(window_size * pr);
  std::vector<float> syn_win = hanning(static_cast<int>(syn_size), 0);

  float syn_sum = 0.0f;
  for (int k = 0; k * out_hop < static_cast<int>(syn_win.size()); k++)
    syn_sum += syn_win[k * out_hop] * syn_win[k * out_hop];
  if (syn_sum == 0.0f)
    syn_sum = 1.0f;

  BiquadLowpass lp(fs, N);
  lp.setCutoff(std::min(nyquist * pr, nyquist * 0.99f));

  std::vector<float> phi(N / 2 + 1, 0.0f);
  std::vector<float> prev_phase(N / 2 + 1, 0.0f);
  std::vector<std::complex<float>> fft_buf(N / 2 + 1);
  std::vector<float> ifft_buf(window_size, 0.0f);
  std::vector<float> resampled_buf(syn_size, 0.0f);

  int wp = 0;
  for (int f = 0; f < static_cast<int>(_fft_cache.size()); f++) {
    if (wp >= static_cast<int>(out.size()))
      break;

    // Copy cached frame and run phase vocoder
    fft_buf = _fft_cache[f];
    for (int i = 0; i < N / 2 + 1; i++) {
      float phase = std::arg(fft_buf[i]);
      float amp = std::abs(fft_buf[i]);
      float expected = 2.0f * M_PI * i * hop / N;
      float delta = phase - prev_phase[i] - expected;
      delta -= 2.0f * M_PI * std::round(delta / (2.0f * M_PI));
      phi[i] += expected + delta;
      phi[i] -= 2.0f * M_PI * std::round(phi[i] / (2.0f * M_PI));
      prev_phase[i] = phase;
      fft_buf[i] = std::polar(amp, phi[i]);
    }

    // IFFT — fftwf_execute_dft_c2r is thread-safe with private buffers
    for (int i = 0; i < N / 2 + 1; i++) {
      fo[i][0] = fft_buf[i].real();
      fo[i][1] = fft_buf[i].imag();
    }
    fftwf_execute_dft_c2r(inv, fo, fi);
    for (int i = 0; i < window_size; i++)
      ifft_buf[i] = fi[i];

    // Cubic resample
    size_t s = ifft_buf.size() - 1;
    size_t L = static_cast<size_t>(std::floor(s * pr));
    if (L < 1)
      L = 1;
    if (resampled_buf.size() != L)
      resampled_buf.resize(L);
    for (size_t i = 0; i < L; i++) {
      float x = static_cast<float>(i) * (s - 1) / (L - 1);
      size_t ix = static_cast<size_t>(std::floor(x));
      float mu = x - ix;
      size_t ix0 = std::min(ix == 0 ? 0 : ix - 1, s);
      size_t ix1 = std::min(ix, s);
      size_t ix2 = std::min(ix + 1 < s ? ix + 1 : s - 1, s);
      size_t ix3 = std::min(ix + 2 < s ? ix + 2 : s - 1, s);
      resampled_buf[i] = lp.process(cubic_interp(
          ifft_buf[ix0], ifft_buf[ix1], ifft_buf[ix2], ifft_buf[ix3], mu));
    }

    // OLA
    for (size_t i = 0;
         i < resampled_buf.size() &&
         (wp + static_cast<int>(i)) < static_cast<int>(out.size());
         i++)
      out[wp + i] += (syn_win[i] * resampled_buf[i]) / (syn_sum * N);

    wp += out_hop;
  }

  return out;
}

void Vocoder::precompute(int min_note, int max_note) {
  _use_precomputed = true;
  int n_notes = max_note - min_note + 1;
  int n_threads = std::min(
      std::max(1, static_cast<int>(std::thread::hardware_concurrency())),
      n_notes);

  std::cout << "[Vocoder] building FFT cache (" << n_notes << " notes, "
            << n_threads << " thread(s))..." << std::endl;
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
        results[t][note] = _synth_note(note, ctx[t].inv, ctx[t].fi, ctx[t].fo);
        std::lock_guard<std::mutex> lock(log_mtx);
        std::cout << "[Vocoder] note " << note << " done" << std::endl;
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

  for (auto &r : results)
    for (auto &[note, buf] : r)
      _precomputed[note] = std::move(buf);

  _fft_cache.clear(); // free ~1.4 MB of cache after precompute
  std::cout << "[Vocoder] precompute done." << std::endl;
}

Vocoder::~Vocoder() {
  fftwf_free(fft_in);
  fftwf_free(fft_out);

  fftwf_destroy_plan(p);
  fftwf_destroy_plan(pi);

  free(file);
}
