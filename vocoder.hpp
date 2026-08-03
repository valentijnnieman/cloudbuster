#pragma once
#include <atomic>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "biquad.hpp"
#include "pa_ringbuffer.h"
#include "sndfile.hh"
#include <fftw3.h>
#include <map>

struct NoteEvent {
  int note;
};

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

  std::atomic<bool> _use_precomputed{false};
  std::map<int, std::vector<float>> _precomputed;

  // --- real-time streaming synthesis state (see stream_sample) ---
  // Fixed-size ring holding stretched-domain OLA output. Power-of-two length;
  // wp/ready/written_end/read_pos are *absolute* (monotonic) positions, wrapped
  // with ring_mask on every access. Sized once by _init_streaming_buffers() in
  // both constructors for the worst-case hop and never resized — no allocation
  // ever happens on the audio thread.
  std::vector<float> stretched;
  size_t ring_mask = 0;
  bool first_frame = true;
  int frame_idx = 0;       // position of the walk through _fft_cache
  int64_t wp = 0;          // next OLA frame start (absolute)
  int64_t ready = 0;       // samples below this are final (absolute)
  int64_t written_end = 0; // high-water mark of zero-then-OLA writes
  // Fractional read position into `stretched` for the streaming pitch
  // resample (advances by 1/pr per output sample). double, not float: it
  // accumulates for the lifetime of a note, and float loses the fraction
  // (audible zipper) on long sustains.
  double read_pos = 0.0;

  // Ceiling on `stretch` in the streaming path — the ring is sized for this
  // worst case, so larger values (only reachable via -s) are clamped.
  static constexpr float STRETCH_MAX = 8.0f;

  // Scratch buffers reused by _synth_one_frame every frame (it runs at hop
  // rate on the audio thread; per-frame vector allocation is not allowed).
  std::vector<std::complex<float>> _fft_scratch; // N/2+1
  std::vector<float> _ifft_scratch;              // N
  std::vector<int> _peaks_scratch;
  std::vector<int> _bounds_scratch;
  // Previous *analysis* frame, kept as complex bins: its phase is only ever
  // needed at the current frame's peak bins, so arg() is computed lazily per
  // peak instead of atan2-ing all N/2+1 bins every frame.
  std::vector<std::complex<float>> _prev_frame;

  // Cache of forward-FFT frames (computed once, reused per note in precompute)
  // Outer index = frame number, inner = N/2+1 complex bins
  std::vector<std::vector<std::complex<float>>> _fft_cache;

  fftwf_plan p;
  fftwf_plan pi;

  float *fft_in;
  fftwf_complex *fft_out;
  float *_fft_in_alloc;
  fftwf_complex *_fft_out_alloc;

public:
  std::vector<float> out_samples;
  int it;
  std::atomic<int> current_note{0};
  // Which note the *MIDI thread* has handed this voice; -1 = never assigned.
  // current_note only catches up once the audio thread drains _note_queue (up
  // to a buffer later), which is too late for the note-on/note-off matching in
  // MidiController — matching on it let a second voice start the same note.
  std::atomic<int> midi_note{-1};
  PaUtilRingBuffer _note_queue;
  NoteEvent _note_queue_buf[16];
  int smpl_ptr;
  std::atomic<bool> running{false};
  std::atomic<bool> stopping{false};
  std::atomic<float> volume{1.0f};
  std::atomic<bool> _cancel_precompute{false};
  bool calculated;

  float adsr_attack = 0.01f;
  float adsr_decay = 0.10f;
  float adsr_sustain = 0.80f;
  float adsr_release = 0.20f;

  int read_ptr = 0;
  int write_ptr = 0;

  bool ROBOTO = false;
  bool WHISPER = false;
  bool ALIEN = false;

  // Pick the synthesis path: false = play back the precomputed note buffers
  // (get_sample), true = synthesize on the audio thread (stream_sample).
  // Carried across the copy constructor so setting it on the template vocoder
  // reaches every voice.
  bool realtime = false;

  float stretch = 1.0f;

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

  /*void calculate_stft_windows();*/

  void calculate_samples(int note);
  float get_sample(int note, int n);
  // Renders the next output sample for `note`, synthesizing frames from
  // _fft_cache on demand. Called per-sample from the audio thread; needs
  // prepare_realtime() to have run first (on the template vocoder, before the
  // voice copies are made — the copy constructor carries _fft_cache along).
  float stream_sample(int note);
  // One-time STFT analysis pass over the loaded file, off the audio thread.
  void prepare_realtime();
  std::vector<float> lowpass_filter(std::vector<float> input, float cutoff);

  void clear();
  void precompute(int min_note, int max_note);
  void apply_precomputed_from(const Vocoder &source);

private:
  enum class AdsrPhase { Attack, Decay, Sustain, Release, Done };
  AdsrPhase _adsr_phase = AdsrPhase::Attack;
  int _adsr_pos = 0;
  float _adsr_level = 0.0f;

  // Runs the ADSR one sample forward. Returns false once the voice is
  // finished (Release elapsed or already Done), in which case `env` is
  // meaningless and the caller must output silence.
  bool _adsr_env(float &env);

  void _build_fft_cache();
  std::vector<float> _synth_note(int note, fftwf_plan inv, float *fi,
                                 fftwf_complex *fo);

  // Sizes the ring + scratch buffers; called from both constructors (off the
  // audio thread).
  void _init_streaming_buffers();
  // Synthesizes one frame into the ring — the streaming counterpart of the
  // per-frame loop body in _synth_note.
  void _synth_one_frame(int note, fftwf_plan inv, float *fi,
                        fftwf_complex *fo);

  // Both fill caller-provided vectors (clear + append) so the per-frame call
  // sites can reuse scratch capacity instead of allocating.
  void get_region_boundaries(const std::vector<int> &peaks, int M,
                             std::vector<int> &out);
  void find_peaks(std::vector<std::complex<float>> &bins,
                  std::vector<int> &peaks);
  void shift_peaks(std::vector<std::complex<float>> &frame,
                   const std::vector<int> &peaks,
                   const std::vector<int> &boundaries, std::vector<float> &phi,
                   std::vector<std::complex<float>> &prev_frame, int hop,
                   float step, bool first_frame);

  std::vector<float> hanning(int N, short itype);
};
