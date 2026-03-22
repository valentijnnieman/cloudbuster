#pragma once
#include "RtMidi.h"
/*#include "sampler.cpp"*/
#include "vocoder.hpp"
#include <thread>

typedef struct {
  SNDFILE *file;
  SF_INFO info;
  std::vector<std::shared_ptr<Vocoder>> voices;
  std::vector<int> indices;
  // std::vector<float> samples;
  std::vector<int> notes;
  int max;
  int index;
  int stln_voice;
} callback_data_s;

class MidiController {
public:
  RtMidiIn *midiIn = 0;
  RtMidiOut *midiOut = 0;

  std::vector<std::string> portNames;

  static void callback(double deltatime, std::vector<unsigned char> *message,
                       void *userData) {
int status = message->at(0) & 0xF0;
int key = message->at(1);
int velocity = message->at(2);

callback_data_s *data = static_cast<callback_data_s *>(userData);

if (status == 0x90 && velocity > 0) { // Note On
    auto& s = data->voices[data->stln_voice];
    s->current_note = key;
    s->clear();
    s->running = true;
    s->stopping = false;
    data->stln_voice = (data->stln_voice + 1) % data->voices.size();
}
if (status == 0x80 || (status == 0x90 && velocity == 0)) { // Note Off
    auto it = std::find_if(data->voices.begin(), data->voices.end(),
        [key](const std::shared_ptr<Vocoder>& s) {
            return s->current_note == key && s->running;
        });
    if (it != data->voices.end()) {
        auto& s = *it;
        if (s) s->stopping = true;
    }
}

  }

  MidiController() {
    try {
      midiIn = new RtMidiIn();
    } catch (RtMidiError &error) {
      std::cout << "Couldn't initialize RtMidiIn: " << std::endl;
      error.printMessage();
    }

    unsigned int nPorts = midiIn->getPortCount();
    std::cout << "\nThere are " << nPorts << " MIDI input sources available.\n";

    std::string portName;

    for (int i = 0; i < nPorts; i++) {
      try {
        portName = midiIn->getPortName(i);
      } catch (RtMidiError &error) {
        error.printMessage();
      }
      std::cout << "  Input Port #" << i + 1 << ": " << portName << '\n';
    }

    try {
      midiOut = new RtMidiOut();
    } catch (RtMidiError &error) {
      error.printMessage();
    }

    nPorts = midiOut->getPortCount();
    std::cout << "\nThere are " << nPorts << " MIDI output ports available.\n";
    for (unsigned int i = 0; i < nPorts; i++) {
      try {
        portName = midiOut->getPortName(i);
        portNames.push_back(portName);
      } catch (RtMidiError &error) {
        error.printMessage();
      }
      std::cout << "  Output Port #" << i + 1 << ": " << portName << '\n';
    }
    std::cout << '\n';
  }

  ~MidiController() {
    delete midiIn;
    delete midiOut;
  }
};
