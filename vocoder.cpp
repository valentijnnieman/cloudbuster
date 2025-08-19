#include "vocoder.hpp"

void Vocoder::print_stats() {
  std::cout << "samples size: " << samples.size();
  std::cout << std::endl;
  // std::cout << "window[0]: " << window[0] << std::endl;
}
Vocoder::Vocoder(const std::string &filename, int N, int window_size, int hop_size_div, float samplerate)
    : it(it), N(N), window_size(window_size), hop_size_div(hop_size_div), calculated(false) {
  /* Open the soundfile */
  file = sf_open(filename.c_str(), SFM_READ, &info);
  samples.resize(info.frames * info.channels);
  sf_readf_float(file, &samples[0], info.frames);
  resampled = std::vector<float>(samples.size(), 0.0f);

  fs = samplerate;
  // downsample to 8k
  double src_ratio = fs / info.samplerate;

  resample((float *)&samples[0], (float *)&resampled[0],
            samples.size() / 2, resampled.size() / 2, src_ratio);

  // done with original samples
  samples.clear();

  /* Close the soundfile */
  sf_close(file);

  pitch_ratio = 1.0f;
  out_samples = std::vector<float>((resampled.size()), 0.0f);
  _buffer = std::vector<float>(window_size, 0.0f);
  _phi = std::vector<float> (N / 2 + 1, 0.0f);
  _fft_buffer = std::vector<std::complex<float>>(N / 2 + 1, 0.0f);

  nyquist = fs / 2.0f;
  cutoff = nyquist / 2.0f;

  sig_len = resampled.size();

  fft_in = (float*)calloc(sig_len, sizeof(float));
  fft_out = (fftwf_complex*)calloc(sig_len, sizeof(fftwf_complex));

  p = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);
  pi = fftwf_plan_dft_c2r_1d(N, fft_out, fft_in, FFTW_ESTIMATE);

  window = hanning(window_size, 0);

  smpl_ptr = 0;
}

void Vocoder::clear()
{
  /*out_samples.clear();*/

  float freq = note_to_freq(current_note);
  float div = 440.0f;
  float r = div / freq;
  pitch_ratio = r;

  out_samples = std::vector<float>((resampled.size() * pitch_ratio), 0.0f);
  calculated = false;
  _calculated_until = 0;
  read_ptr = 0;
  write_ptr = 0;
  smpl_ptr = 0;
}

std::vector<float> Vocoder::hanning(int N, short itype)
{
    int half, i, idx, n;
    std::vector<float> w(N, 0.0f);

    // w = (float*) calloc(N, sizeof(float));
    // memset(w, 0, N*sizeof(float));

    if(itype==1)    //periodic function
        n = N-1;
    else
        n = N;

    if(n%2==0)
    {
        half = n/2;
        for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
            w[i] = 0.5 * (1 - cos(2*M_PI*(i+1) / (n+1)));

        idx = half-1;
        for(i=half; i<n; i++) {
            w[i] = w[idx];
            idx--;
        }
    }
    else
    {
        half = (n+1)/2;
        for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
            w[i] = 0.5 * (1 - cos(2*M_PI*(i+1) / (n+1)));

        idx = half-2;
        for(i=half; i<n; i++) {
            w[i] = w[idx];
            idx--;
        }
    }

    if(itype==1)    //periodic function
    {
        for(i=N-1; i>=1; i--)
            w[i] = w[i-1];
        w[0] = 0.0;
    }
    return(w);
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
  fftwf_execute_dft_r2c(pi, fft_in, fft_out);
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

std::vector<float> Vocoder::lowpass_filter(std::vector<float> input, float cutoff) {
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
  /*float freq = note_to_freq(note);*/
  /*float div = 440.0f;*/
  /*float r = div / freq;*/
  /*pitch_ratio = r;*/

  if (n >= _calculated_until) {
    // Prepare new window
    size_t resampled_size = static_cast<size_t>(window_size * pitch_ratio);
    if (_resampled.size() != resampled_size)
      _resampled.resize(resampled_size, 0.0f);

    synthesis_hopsize = window_size / hop_size_div;
    analysis_hopsize = synthesis_hopsize * pitch_ratio;

    // Windowed samples
    for (size_t i = 0; i < window_size; i++) {
      size_t sample_idx = read_ptr + i;
      _buffer[i] = (sample_idx < resampled.size() ? window[i] * resampled[sample_idx] : 0.0f);
    }

    auto filtered_buffer = lowpass_filter(_buffer, (_fft_buffer.size() * pitch_ratio) / 2);
    // FFT
    forward_fft(filtered_buffer.data(), reinterpret_cast<std::complex<float>*>(_fft_buffer.data()));

    // Phase vocoder processing
    for (size_t i = 0; i < _fft_buffer.size(); i++) {
      float phase = std::arg(_fft_buffer[i]);
      float previous_phase = (i > 0) ? std::arg(_fft_buffer[i - 1]) : 0.0f;
      float amplitude = std::abs(_fft_buffer[i]);
      float freq_bin = 2.0f * M_PI * static_cast<float>(i) / _fft_buffer.size();

      float target = previous_phase + (freq_bin * analysis_hopsize);
      float deviation = phase - target;
      float increment = (freq_bin * analysis_hopsize) + deviation;

      if (INSTANTANEOUS) {
        float fi = increment / (2 * M_PI * analysis_hopsize) * fs;
        freq_bin = fi;
      }
      float delta_phi = (freq_bin * analysis_hopsize) +
                        std::arg(phase - previous_phase - (freq_bin * analysis_hopsize));
      if (GONGO) {
        delta_phi = (freq_bin * analysis_hopsize) + phase - previous_phase -
                    (freq_bin * analysis_hopsize);
      }

      if (PHI_UNWRAP) {
        _phi[i] = std::arg(delta_phi * synthesis_hopsize);
      } else {
        _phi[i] = delta_phi * synthesis_hopsize;
      }

      _fft_buffer[i].real(amplitude * std::cos(_phi[i]));
      _fft_buffer[i].imag(amplitude * std::sin(_phi[i]));
    }

    // Inverse FFT
    ifft(_buffer.data(), reinterpret_cast<std::complex<float>*>(_fft_buffer.data()));

    // Resample
    size_t s = _fft_buffer.size();
    size_t L = static_cast<size_t>(std::floor(s * pitch_ratio));
    for (size_t i = 0; i < L; i++) {
      /*float x = static_cast<float>(i) * s / L;*/
      float x = static_cast<float>(i) * (s - 1) / (L - 1);
      size_t ix = static_cast<size_t>(std::floor(x));
      float mu = x - ix;
      // Clamp indices to valid range
      size_t ix0 = (ix == 0) ? 0 : ix - 1;
      size_t ix1 = ix;
      size_t ix2 = (ix + 1 < s) ? ix + 1 : s - 1;
      size_t ix3 = (ix + 2 < s) ? ix + 2 : s - 1;
      _resampled[i] = cubic_interp(_buffer[ix0], _buffer[ix1], _buffer[ix2], _buffer[ix3], mu);
    }

    // low pass filter
    /*auto filtered_buffer = lowpass_filter(_resampled, fs / 4);*/

    // Overlap-add output with normalization
    float window_sum = 0.0f;
    for (size_t i = 0; i < window_size; i++) {
      window_sum += window[i];
    }
    for (size_t i = 0; i < window_size && (read_ptr + i) < out_samples.size(); i++) {
      out_samples[read_ptr + i] += (window[i] * _resampled[i]) / window_sum;
    }

    // Increment pointers
    read_ptr += static_cast<size_t>(analysis_hopsize);
    write_ptr += static_cast<size_t>(synthesis_hopsize);
    _calculated_until += static_cast<size_t>(analysis_hopsize);

    if (read_ptr >= resampled.size() - window_size) {
      read_ptr = 0;
      write_ptr = 0;
      calculated = true;
    }
  }

  if (n >= out_samples.size() || n >= _calculated_until) {
    running = false;
    current_note = 0;
    stopping = false;
    smpl_ptr = 0;
    return 0.0f;
  }

  return out_samples[n] * gain;
}


Vocoder::~Vocoder() {
  fftwf_destroy_plan(p);
  fftwf_destroy_plan(pi);
  free(file);
}
