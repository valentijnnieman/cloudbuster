#pragma once
#include <atomic>
#include <complex>
#include <iostream>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "biquad.hpp"
#include "sndfile.hh"
#include <fftw3.h>
#include <map>
#include <samplerate.h>

class Vocoder {
private:
  SNDFILE *file;
  SF_INFO info;

  int _calculated_until = 0;
  int sig_len, analysis_hopsize, synthesis_hopsize;
  int N;
  int window_size;
  int hop_size_div;
  float pitch_ratio;
  float gain = 1.0f;

  float nyquist;

  std::vector<float> samples;
  std::vector<float> resampled;

  std::vector<float> _resampled;
  std::vector<float> _buffer;
  std::vector<float> _phi;
  std::vector<float> _prev_phase;
  std::vector<std::complex<float>> _fft_buffer;

  BiquadLowpass lowpass; // cutoff 5 kHz, Q = 0.707 (Butterworth)

  float cutoff;

  std::vector<float> window;
  std::vector<float> _syn_window;
  float _syn_window_sum = 0.0f;

  bool _use_precomputed = false;
  std::map<int, std::vector<float>> _precomputed;

  // Cache of forward-FFT frames (computed once, reused per note in precompute)
  // Outer index = frame number, inner = N/2+1 complex bins
  std::vector<std::vector<std::complex<float>>> _fft_cache;

  fftwf_plan p;
  fftwf_plan pi;

  float *fft_in;
  fftwf_complex *fft_out;

public:
  std::vector<float> out_samples;
  int it;
  std::atomic<int> current_note{0};
  std::atomic<int> pending_note{0};
  std::atomic<bool> clear_pending{false};
  int smpl_ptr;
  std::atomic<bool> running{false};
  std::atomic<bool> stopping{false};
  bool calculated;

  int read_ptr = 0;
  int write_ptr = 0;

  bool PHI_UNWRAP;
  bool GONGO;
  bool INSTANTANEOUS;

  float fs;

  int frames_size() { return resampled.size(); }

  void print_stats();
  Vocoder(const std::string &filename, int N = 1024, int window_size = 1024,
          int hop_size_div = 4, float samplerate = 44100.0f);
  Vocoder(const Vocoder &other);
  ~Vocoder();

  std::vector<float> &get_samples(int frameCount, float amp);

  float note_to_freq(int note);

  float cubic_interp(float y0, float y1, float y2, float y3, float mu);

  void forward_fft(float *time_data, std::complex<float> *freq_data);

  void ifft(float *time_data, std::complex<float> *freq_data);

  void resample(float *data_in, float *data_out, int input_size,
                int output_size, float pitch_ratio);

  /*void calculate_stft_windows();*/

  void calculate_samples(int note);
  float get_sample(int note, int n);
  std::vector<float> lowpass_filter(std::vector<float> input, float cutoff);

  void clear();
  void precompute(int min_note, int max_note);

private:
  void _build_fft_cache();
  std::vector<float> _synth_note(int note, fftwf_plan inv,
                                  float *fi, fftwf_complex *fo);

  std::vector<float> hanning(int N, short itype);
};
