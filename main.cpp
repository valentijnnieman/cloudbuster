#include <algorithm>
#include <complex>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <string.h>

#include "sndfile.hh"
#include <samplerate.h>
#include <vector>

#include "portaudio.h"

#include "midi_controller.hpp"
#include "vocoder.hpp"
#include <cmath>
#include <numbers>

// #include <matplot/matplot.h>
// using namespace matplot;

PaStream *stream;

static int callback(const void *input, void *output, unsigned long frameCount,
                    const PaStreamCallbackTimeInfo *timeInfo,
                    PaStreamCallbackFlags statusFlags, void *userData) {
  callback_data_s *data = (callback_data_s *)userData;
  float *out = (float *)output;
  std::fill_n(out, frameCount * 2, 0.0f);  // stereo: 2 samples per frame

  if (data->reloading.load()) return paContinue;

  uint8_t amplitudes = 0;
  for (auto &sampler : data->voices) {
    if (sampler->clear_pending.exchange(false)) {
      sampler->current_note.store(sampler->pending_note.load());
      sampler->clear();
      sampler->running.store(true);
    }

    if (sampler->running.load()) {
      amplitudes++;
      int note = sampler->current_note.load();
      for (size_t i = 0; i < frameCount; i++) {
        float s = sampler->get_sample(note, sampler->smpl_ptr + i);
        out[i * 2]     += s;  // left
        out[i * 2 + 1] += s;  // right
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

void close_stream(int s) {
  printf("Closing...\n");
  PaError error;
  /*  Shut down portaudio */
  error = Pa_CloseStream(stream);
  if (error != paNoError) {
    fprintf(stderr, "Problem closing stream\n");
  }

  error = Pa_Terminate();
  if (error != paNoError) {
    fprintf(stderr, "Problem terminating\n");
  }
  exit(1);
}

int main(int argc, const char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);

  bool pv = false;
  bool stft = false;
  bool phi_unwrap = true;
  bool precompute_flag = true;
  bool gongo = false;
  bool instantaneous = false;
  int N = 1024;
  int window_size = N;
  float fs = 8000.0;
  int min_note = 40;
  int max_note = 90;
  int hop_size_div = 8;
  int midi_in = 1;
  int midi_out = 1;
  int device = 0;
  std::string folder = ".";

  for (int i = 0; i < args.size(); i++) {
    if (args[i] == "-f") {
      folder = args[i + 1];
      std::cout << "[Sampler] using folder: " << folder << std::endl;
    }
    if (args[i] == "-n") {
      N = stoi(args[i + 1]);
      std::cout << "[Sampler] N (fft) size: " << N << std::endl;
      window_size = stoi(args[i + 1]);
      std::cout << "[Sampler] window size: " << window_size << std::endl;
    }
    if (args[i] == "-h") {
      hop_size_div = stoi(args[i + 1]);
      std::cout << "[Sampler] hop size divide by: " << hop_size_div
                << std::endl;
    }
    if (args[i] == "-fs") {
      fs = stof(args[i + 1]);
      std::cout << "[Sampler] sample rate: " << fs << std::endl;
    }
    if (args[i] == "-min") {
      min_note = stoi(args[i + 1]);
      std::cout << "[Sampler] minimum (midi) note: " << min_note << std::endl;
    }
    if (args[i] == "-max") {
      max_note = stoi(args[i + 1]);
      std::cout << "[Sampler] maximum (midi) note: " << max_note << std::endl;
    }
    if (args[i] == "-midi") {
      midi_in = stoi(args[i + 1]);
      midi_out = midi_in;
      std::cout << "[Sampler] midi i/o ports: " << midi_in << std::endl;
    }
    if (args[i] == "-device") {
      device = stoi(args[i + 1]);
      std::cout << "[Sampler] device: " << device << std::endl;
    } else {
      if (args[i] == "no-unwrap") {
        phi_unwrap = false;
        std::cout << "[Sampler] not unwrapping phi values in algorithm"
                  << std::endl;
      }
      if (args[i] == "gongo") {
        gongo = true;
        std::cout << "[Sampler] using GONGO preset" << std::endl;
      }
      if (args[i] == "instantaneous") {
        instantaneous = true;
        std::cout << "[Sampler] calculate instantaneous frequency in algorithm"
                  << std::endl;
      }
      if (args[i] == "no-precompute") {
        precompute_flag = false;
        std::cout << "[Sampler] precompute disabled" << std::endl;
      }
      if (args[i] == "pitchshift") {
        pv = true;
        std::cout << "[Sampler] using phase-vocoder pitch shifting algorithm"
                  << std::endl;
      }
      if (args[i] == "phase-vocoder") {
        pv = false;
        stft = true;
        std::cout << "[Sampler] using phase vocoder algorithm" << std::endl;
      }
    }
  }

  // Scan folder for audio files
  std::vector<std::string> file_list;
  {
    namespace fs = std::filesystem;
    for (auto &entry : fs::directory_iterator(folder)) {
      if (!entry.is_regular_file()) continue;
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".wav" || ext == ".aif" || ext == ".aiff" || ext == ".flac" || ext == ".ogg") {
        file_list.push_back(entry.path().string());
      }
    }
    std::sort(file_list.begin(), file_list.end());
  }
  if (file_list.empty()) {
    std::cerr << "[Sampler] no audio files found in: " << folder << std::endl;
    return 1;
  }
  std::cout << "[Sampler] found " << file_list.size() << " file(s):" << std::endl;
  for (auto &f : file_list) std::cout << "  " << f << std::endl;

  std::string filename = file_list[0];
  std::cout << "[Sampler] loading: " << filename << std::endl;

  Vocoder sampler = Vocoder(filename, N, window_size, hop_size_div, fs);
  sampler.PHI_UNWRAP = phi_unwrap;
  sampler.GONGO = gongo;
  sampler.INSTANTANEOUS = instantaneous;

  if (precompute_flag)
    sampler.precompute(min_note, max_note);

  PaError error;
  callback_data_s data;

  data.stln_voice = 0;
  data.file_list = file_list;
  data.current_file_index.store(0);

  MidiController ctrl;

  for (auto &name : ctrl.portNames) {
    std::cout << name << std::endl;
  }

  ctrl.midiIn->openPort(midi_in);
  ctrl.midiIn->setCallback(ctrl.callback, &data);

  ctrl.midiOut->openPort(midi_out);

  signal(SIGINT, close_stream);

  std::cout << "copying samplers for voices..." << std::endl;
  for (int j = 0; j < 4; j++) {
    Vocoder *s = new Vocoder(sampler);
    s->it = j;

    data.voices.push_back(std::shared_ptr<Vocoder>(s));
  }

  /* init portaudio */
  error = Pa_Initialize();
  if (error != paNoError) {
    fprintf(stderr, "Problem initializing\n");
    return 1;
  }
  PaStreamParameters outputParameters;

  int numDevices = Pa_GetDeviceCount();
  if (numDevices < 0) {
    printf("ERROR: Pa_GetDeviceCount returned 0x%x\n", numDevices);
  }
  const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(device);
  printf("Name                        = %s\n", deviceInfo->name);
  printf("Host API                    = %s\n",
         Pa_GetHostApiInfo(deviceInfo->hostApi)->name);
  printf("Max inputs = %d", deviceInfo->maxInputChannels);
  printf(", Max outputs = %d\n", deviceInfo->maxOutputChannels);

  printf("Default low input latency   = %8.4f\n",
         deviceInfo->defaultLowInputLatency);
  printf("Default low output latency  = %8.4f\n",
         deviceInfo->defaultLowOutputLatency);
  printf("Default high input latency  = %8.4f\n",
         deviceInfo->defaultHighInputLatency);
  printf("Default high output latency = %8.4f\n",
         deviceInfo->defaultHighOutputLatency);

  outputParameters.device = device;
  if (outputParameters.device == paNoDevice) {
    fprintf(stderr, "Error: No default output device.\n");
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
    fprintf(stderr, "Problem opening Default Stream\n");
    return 1;
  }

  /* Start the stream */
  error = Pa_StartStream(stream);
  if (error != paNoError) {
    fprintf(stderr, "Problem opening starting Stream\n");
    return 1;
  }

  std::cout << "Done! opening stream..." << std::endl;
  /* Run until EOF is reached */
  while (Pa_IsStreamActive(stream)) {
    if (data.file_change_pending.exchange(false)) {
      int idx = data.pending_file_index.load();
      std::cout << "[Sampler] switching to (" << idx + 1 << "/"
                << data.file_list.size() << "): " << data.file_list[idx]
                << std::endl;

      data.reloading.store(true);
      Pa_Sleep(50); // let any in-flight callback finish

      Vocoder new_sampler(data.file_list[idx], N, window_size, hop_size_div, fs);
      new_sampler.PHI_UNWRAP = phi_unwrap;
      new_sampler.GONGO = gongo;
      new_sampler.INSTANTANEOUS = instantaneous;
      if (precompute_flag) new_sampler.precompute(min_note, max_note);

      data.voices.clear();
      for (int j = 0; j < 4; j++) {
        Vocoder *s = new Vocoder(new_sampler);
        s->it = j;
        data.voices.push_back(std::shared_ptr<Vocoder>(s));
      }
      data.stln_voice = 0;
      data.current_file_index.store(idx);
      data.reloading.store(false);
    }
    Pa_Sleep(100);
  }

  close_stream(0);

  return 0;
}
