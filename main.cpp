#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <string.h>

#include <vector>
#include <samplerate.h>
#include "sndfile.hh"

#include "portaudio.h"

#include "vocoder.hpp"
#include "midi_controller.hpp"
#include <cmath>
#include <numbers>

// #include <matplot/matplot.h>
// using namespace matplot;

PaStream *stream;
const float amp = 0.5f;

static int callback(const void *input, void *output, unsigned long frameCount,
                    const PaStreamCallbackTimeInfo *timeInfo,
                    PaStreamCallbackFlags statusFlags, void *userData) {
  frameCount *= 2;
  callback_data_s *data = (callback_data_s *)userData;
  float *out = (float *)output;
  memset(out, 0.0f, sizeof(float) * frameCount);

  for (const auto& sampler : data->voices) {
    if (sampler->running && !sampler->stopping) {
      for (size_t i = 0; i < frameCount; i++) {
        out[i] += sampler->get_sample(sampler->current_note, sampler->smpl_ptr+i);
      }
    }
    if (sampler->stopping) {
      for (size_t i = 0; i < frameCount; i++) {
        out[i] += sampler->get_sample(sampler->current_note, sampler->smpl_ptr+i);
      }
    }

    sampler->smpl_ptr += frameCount;
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
  bool gongo = false;
  bool instantaneous = false;
  int N = 1024;
  int window_size = 1024;
  float fs = 8000.0;
  int min_note = 40;
  int max_note = 90;
  int hop_size_div = 8;
  int midi_in = 1;
  int midi_out = 1;
  int device = 0;
  std::string filename =
      "/home/valentijn/dev/c++/cloudbuster/fairchild/samples/Piano_note_a1.wav";

  for (int i = 0; i < args.size(); i++) {
    if (args[i] == "-f") {
      filename = args[i + 1];
      std::cout << "[Sampler] using file: " << filename << std::endl;
    }
    if (args[i] == "-n") {
      N = stoi(args[i + 1]);
      std::cout << "[Sampler] N (fft) size: " << N << std::endl;
    }
    if (args[i] == "-w") {
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
      std::cout << "[Sampler] sample rate: " << fs
                << std::endl;
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

  // SndfileHandle file;
  /*Sampler sampler = Sampler(filename, N, window_size, hop_size_div);*/
  Vocoder sampler = Vocoder(filename, N, window_size, hop_size_div, fs);
  sampler.PHI_UNWRAP = phi_unwrap;
  sampler.GONGO = gongo;
  sampler.INSTANTANEOUS = instantaneous;

  PaError error;
  callback_data_s data;

  data.stln_voice = 0;

  MidiController ctrl;

  for (auto &name : ctrl.portNames) {
    std::cout << name << std::endl;
  }

  ctrl.midiIn->openPort(midi_in);
  ctrl.midiIn->setCallback(ctrl.callback, &data);

  ctrl.midiOut->openPort(midi_out);

  signal(SIGINT, close_stream);

  std::cout << "copying samplers for voices..." << std::endl;
  for (int j = 0; j < 8; j++) {
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

  /*std::cout << "calculating samples..." << std::endl;*/
  /*for (int i = min_note; i < max_note; i++) {*/
  /*  if (pv) {*/
  /*    sampler.calculate_sample_pitch_shift(data, i);*/
  /*  }*/
  /*  if (!pv && stft) {*/
  /*    sampler.calculate_sample_stft(data, i);*/
  /*  }*/
  /*  if (!pv && !stft) {*/
  /*    sampler.calculate_sample(data, i);*/
  /*  }*/

    // copy samples to all voices
    /*for (auto s : data.voices) {*/
    /*  s->key_samples[i] = sampler.key_samples[i];*/
    /*}*/
  /*}*/

  std::cout << "Done! opening stream..." << std::endl;
  /* Run until EOF is reached */
  while (Pa_IsStreamActive(stream)) {
    // Pa_Sleep(1000);
  }

  close_stream(0);

  return 0;
}
