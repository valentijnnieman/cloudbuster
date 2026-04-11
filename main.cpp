#include <algorithm>
#include <chrono>
#include <complex>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <string.h>
#include <thread>

#include "sndfile.hh"
#include <vector>

#include "portaudio.h"

#include "midi_controller.hpp"
#include "vocoder.hpp"
#include <cmath>
#include <numbers>

// #include <matplot/matplot.h>
// using namespace matplot;

PaStream *stream;
static volatile sig_atomic_t keep_running = 1;

static void handle_signal(int) { keep_running = 0; }

static int callback(const void *input, void *output, unsigned long frameCount,
                    const PaStreamCallbackTimeInfo *timeInfo,
                    PaStreamCallbackFlags statusFlags, void *userData) {
  callback_data_s *data = (callback_data_s *)userData;
  float *out = (float *)output;
  std::fill_n(out, frameCount * 2, 0.0f); // stereo: 2 samples per frame

  if (data->reloading.load())
    return paContinue;

  uint8_t amplitudes = 0;
  for (auto &sampler : data->voices) {
    NoteEvent ev;
    while (PaUtil_ReadRingBuffer(&sampler->_note_queue, &ev, 1)) {
      sampler->current_note.store(ev.note);
      sampler->clear();
      sampler->running.store(true);
    }

    if (sampler->running.load()) {
      amplitudes++;
      int note = sampler->current_note.load();
      for (size_t i = 0; i < frameCount; i++) {
        float s = sampler->get_sample(note, sampler->smpl_ptr + i);
        out[i * 2] += s;     // left
        out[i * 2 + 1] += s; // right
      }
      sampler->smpl_ptr += frameCount;
    }
  }

  if (amplitudes > 0) {
    for (size_t i = 0; i < frameCount * 2; ++i) {
      out[i] = std::clamp(out[i] / amplitudes, -1.0f, 1.0f);
    }
  }

  return paContinue;
}

static void close_stream() {
  std::cout << "closing..." << std::endl;
  PaError error;

  error = Pa_StopStream(stream);
  if (error != paNoError)
    std::cerr << "stop: " << Pa_GetErrorText(error) << std::endl;

  error = Pa_CloseStream(stream);
  if (error != paNoError)
    std::cerr << "close: " << Pa_GetErrorText(error) << std::endl;

  error = Pa_Terminate();
  if (error != paNoError)
    std::cerr << "term: " << Pa_GetErrorText(error) << std::endl;
}

int main(int argc, const char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);

  bool pv = false;
  bool stft = false;
  bool precompute_flag = true;
  bool roboto = false;
  bool whisper = false;
  bool alien = false;
  int N = 1024;
  int window_size = N;
  float fs = 8000.0;
  int min_note = 40;
  int max_note = 90;
  int hop_size_div = 8;
  float stretch = 2.0f;
  int midi_in = 1;
  int midi_out = 1;
  int device = 0;
  int num_voices = 4;
  std::string folder = ".";

  for (int i = 0; i < args.size(); i++) {
    if (args[i] == "-f") {
      folder = args[i + 1];
      std::cout << "dir: " << folder << std::endl;
    }
    if (args[i] == "-n") {
      N = stoi(args[i + 1]);
      std::cout << "N=" << N << std::endl;
      window_size = stoi(args[i + 1]);
      std::cout << "win=" << window_size << std::endl;
    }
    if (args[i] == "-h") {
      hop_size_div = stoi(args[i + 1]);
      std::cout << "hop/=" << hop_size_div << std::endl;
    }
    if (args[i] == "-fs") {
      fs = stof(args[i + 1]);
      std::cout << "fs=" << fs << std::endl;
    }
    if (args[i] == "-s") {
      stretch = stof(args[i + 1]);
      std::cout << "stretch=" << stretch << std::endl;
    }
    if (args[i] == "-min") {
      min_note = stoi(args[i + 1]);
      std::cout << "min=" << min_note << std::endl;
    }
    if (args[i] == "-max") {
      max_note = stoi(args[i + 1]);
      std::cout << "max=" << max_note << std::endl;
    }
    if (args[i] == "-midi") {
      midi_in = stoi(args[i + 1]);
      midi_out = midi_in;
      std::cout << "midi=" << midi_in << std::endl;
    }
    if (args[i] == "-device") {
      device = stoi(args[i + 1]);
      std::cout << "device=" << device << std::endl;
    } else {
      if (args[i] == "roboto") {
        roboto = true;
        std::cout << "roboto" << std::endl;
      }
      if (args[i] == "whisper") {
        whisper = true;
        std::cout << "whisper" << std::endl;
      }
      if (args[i] == "alien") {
        alien = true;
        std::cout << "alien" << std::endl;
      }
      if (args[i] == "no-precompute") {
        precompute_flag = false;
        std::cout << "no precompute" << std::endl;
      }
    }
  }

  // Scan folder for audio files
  std::vector<std::string> file_list;
  {
    namespace fs = std::filesystem;
    for (auto &entry : fs::directory_iterator(folder)) {
      if (!entry.is_regular_file())
        continue;
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".wav" || ext == ".aif" || ext == ".aiff" || ext == ".flac" ||
          ext == ".ogg") {
        file_list.push_back(entry.path().string());
      }
    }
    std::sort(file_list.begin(), file_list.end());
  }
  if (file_list.empty()) {
    std::cerr << "no files in: " << folder << std::endl;
    return 1;
  }
  std::cout << "found " << file_list.size() << " files" << std::endl;
  for (auto &f : file_list)
    std::cout << std::filesystem::path(f).filename().string() << std::endl;

  std::string filename = file_list[0];
  std::cout << "loading: "
            << std::filesystem::path(filename).filename().string() << std::endl;

  Vocoder sampler = Vocoder(filename, N, window_size, hop_size_div, fs);
  sampler.ROBOTO = roboto;
  sampler.WHISPER = whisper;
  sampler.ALIEN = alien;
  sampler.stretch = stretch;

  PaError error;
  callback_data_s data;

  data.stln_voice = 0;
  data.file_list = file_list;
  data.current_file_index.store(0);

  data.roboto.store(roboto);
  data.whisper.store(whisper);
  data.alien.store(alien);

  data.pending_N.store(N);
  data.pending_hop_size_div.store(hop_size_div);
  data.pending_stretch.store(stretch);

  MidiController ctrl;

  for (auto &name : ctrl.portNames) {
    std::cout << name << std::endl;
  }

  ctrl.midiIn->openPort(midi_in);
  ctrl.midiIn->setCallback(ctrl.callback, &data);

  ctrl.midiOut->openPort(midi_out);

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);

  std::cout << "init voices..." << std::endl;
  for (int j = 0; j < num_voices; j++) {
    Vocoder *s = new Vocoder(sampler);
    s->it = j;

    data.voices.push_back(std::shared_ptr<Vocoder>(s));
  }

  /* init portaudio */
  error = Pa_Initialize();
  if (error != paNoError) {
    std::cerr << "PA init failed" << std::endl;
    return 1;
  }
  PaStreamParameters outputParameters;

  int numDevices = Pa_GetDeviceCount();
  std::cout << "PA devs: " << numDevices << std::endl;

  const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(device);
  if (deviceInfo == nullptr) {
    std::cerr << "dev " << device << " not found" << std::endl;
    Pa_Terminate();
    return 1;
  }
  std::cout << "dev: " << deviceInfo->name << std::endl;
  const PaHostApiInfo *hostApi = Pa_GetHostApiInfo(deviceInfo->hostApi);
  std::cout << "API: " << (hostApi ? hostApi->name : "unknown") << std::endl;
  std::cout << "in:" << deviceInfo->maxInputChannels
            << " out:" << deviceInfo->maxOutputChannels << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "llo in: " << deviceInfo->defaultLowInputLatency << std::endl;
  std::cout << "llo out:" << deviceInfo->defaultLowOutputLatency << std::endl;
  std::cout << "lhi in: " << deviceInfo->defaultHighInputLatency << std::endl;
  std::cout << "lhi out:" << deviceInfo->defaultHighOutputLatency << std::endl;

  outputParameters.device = device;
  if (outputParameters.device == paNoDevice) {
    std::cerr << "no output device" << std::endl;
  }
  outputParameters.channelCount = 2;
  outputParameters.sampleFormat = paFloat32;
  outputParameters.suggestedLatency =
      Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  error =
      Pa_OpenStream(&stream, NULL, &outputParameters, /* &outputParameters, */
                    fs, 256, paNoFlag, callback, &data);

  if (error != paNoError) {
    std::cerr << "stream open failed" << std::endl;
    return 1;
  }

  /* Start the stream */
  error = Pa_StartStream(stream);
  if (error != paNoError) {
    std::cerr << "stream start failed" << std::endl;
    return 1;
  }

  std::unique_ptr<Vocoder> precompute_tmpl;
  std::thread precompute_thread;

  auto launch_precompute = [&](const std::string &filename) {
    precompute_tmpl =
        std::make_unique<Vocoder>(filename, N, window_size, hop_size_div, fs);
    precompute_tmpl->ROBOTO = data.roboto.load();
    precompute_tmpl->WHISPER = data.whisper.load();
    precompute_tmpl->ALIEN = data.alien.load();
    precompute_tmpl->stretch = stretch;
    precompute_thread = std::thread(
        [&, fname = std::filesystem::path(filename).filename().string()]() {
          precompute_tmpl->precompute(min_note, max_note);
          if (!precompute_tmpl->_cancel_precompute.load()) {
            data.reloading.store(true);
            Pa_Sleep(50);
            for (auto &v : data.voices)
              v->apply_precomputed_from(*precompute_tmpl);
            data.reloading.store(false);
            MidiController::print_with_filename(&data, "done!");
          }
        });
  };

  if (precompute_flag)
    launch_precompute(file_list[0]);

  // std::cout << "Done! opening stream..." << std::endl;
  /* Run until stream ends or signal received */
  while (keep_running && Pa_IsStreamActive(stream)) {
    if (data.file_change_pending.exchange(false)) {
      int idx = data.pending_file_index.load();
      // Cancel any ongoing precompute and wait for it to finish
      if (precompute_tmpl)
        precompute_tmpl->_cancel_precompute.store(true);
      if (precompute_thread.joinable())
        precompute_thread.join();

      // Swap voices immediately with real-time vocoder (no precomputed data
      // yet)
      data.reloading.store(true);
      Pa_Sleep(50);

      Vocoder new_sampler(data.file_list[idx], N, window_size, hop_size_div,
                          fs);
      new_sampler.ROBOTO = data.roboto;
      new_sampler.WHISPER = data.whisper;
      new_sampler.ALIEN = data.alien;
      new_sampler.stretch = stretch;
      data.voices.clear();
      for (int j = 0; j < num_voices; j++) {
        Vocoder *s = new Vocoder(new_sampler);
        s->it = j;
        data.voices.push_back(std::shared_ptr<Vocoder>(s));
      }
      data.stln_voice = 0;
      data.current_file_index.store(idx);
      data.reloading.store(false);

      // Start background precompute for the new file
      if (precompute_flag)
        launch_precompute(data.file_list[idx]);
    }

    constexpr int64_t PARAM_DEBOUNCE_MS = 500;
    if (data.param_change_dirty.load() &&
        MidiController::now_ms() - data.param_last_change_ms.load() >= PARAM_DEBOUNCE_MS) {
      data.param_change_dirty.store(false);
      N = data.pending_N.load();
      window_size = N;
      hop_size_div = data.pending_hop_size_div.load();
      stretch = data.pending_stretch.load();

      if (precompute_tmpl)
        precompute_tmpl->_cancel_precompute.store(true);
      if (precompute_thread.joinable())
        precompute_thread.join();

      data.reloading.store(true);
      Pa_Sleep(50);
      int idx = data.current_file_index.load();
      Vocoder new_sampler(data.file_list[idx], N, window_size, hop_size_div,
                          fs);
      new_sampler.ROBOTO = data.roboto;
      new_sampler.WHISPER = data.whisper;
      new_sampler.ALIEN = data.alien;
      new_sampler.stretch = stretch;
      data.voices.clear();
      for (int j = 0; j < num_voices; j++) {
        Vocoder *s = new Vocoder(new_sampler);
        s->it = j;
        data.voices.push_back(std::shared_ptr<Vocoder>(s));
      }
      data.stln_voice = 0;
      data.reloading.store(false);

      if (precompute_flag)
        launch_precompute(data.file_list[idx]);
    }

    Pa_Sleep(100);
  }

  // Cancel and join precompute thread before shutdown
  if (precompute_tmpl)
    precompute_tmpl->_cancel_precompute.store(true);
  if (precompute_thread.joinable())
    precompute_thread.join();

  close_stream();

  return 0;
}
