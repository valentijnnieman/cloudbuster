#pragma once
#include "RtMidi.h"
/*#include "sampler.cpp"*/
#include "vocoder.hpp"
#include <chrono>
#include <iomanip>
#include <string>
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

  // multi-file navigation
  std::vector<std::string> file_list;
  std::atomic<int> current_file_index{0};
  std::atomic<int> pending_file_index{0};
  std::atomic<bool> file_change_pending{false};
  std::atomic<bool> reloading{false};

  std::atomic<bool> roboto{false};
  std::atomic<bool> whisper{false};
  std::atomic<bool> alien{false};
} callback_data_s;

class MidiController {
public:
  RtMidiIn *midiIn = 0;
  RtMidiOut *midiOut = 0;

  std::vector<std::string> portNames;

  static void print_with_filename(callback_data_s *data,
                                  const std::string &msg) {
    std::cout << std::filesystem::path(
                     data->file_list[data->current_file_index])
                     .filename()
                     .string()
              << std::endl;
    std::cout << msg << std::endl;
  }

  static void callback(double deltatime, std::vector<unsigned char> *message,
                       void *userData) {
    if (message->size() < 3)
      return;
    int status = message->at(0) & 0xF0;
    int key = message->at(1);
    int velocity = message->at(2);

    callback_data_s *data = static_cast<callback_data_s *>(userData);

    if (status == 0x90 && velocity > 0) { // Note On
      auto &s = data->voices[data->stln_voice];
      s->pending_note.store(key);
      s->stopping.store(false);
      s->clear_pending.store(
          true); // audio thread will call clear() + set running
      data->stln_voice = (data->stln_voice + 1) % data->voices.size();
    }
    if (status == 0x80 || (status == 0x90 && velocity == 0)) { // Note Off
      auto it = std::find_if(data->voices.begin(), data->voices.end(),
                             [key](const std::shared_ptr<Vocoder> &s) {
                               return s->current_note.load() == key &&
                                      s->running.load();
                             });
      if (it != data->voices.end()) {
        (*it)->stopping.store(true);
      }
    }
    if (status == 0xB0) { // Control Change
      if (key == 7) {     // CC7: Channel Volume
        float vol = velocity / 127.0f;
        for (auto &v : data->voices)
          v->volume.store(vol);
        std::string msg = "vol: " + std::to_string(vol);
        print_with_filename(data, msg);
      }
      int n = (int)data->file_list.size();
      if (n > 1) {
        // ADSR: CC73=Attack, CC75=Decay, CC79=Sustain level, CC72=Release
        if (key == 73) { // Attack time: 0–2 s
          float val = std::max(0.001f, (velocity / 127.0f) * 2.0f);
          for (auto &v : data->voices)
            v->adsr_attack = val;
          std::string msg = "atk: " + std::to_string(val) + "s";
          print_with_filename(data, msg);
        }
        if (key == 75) { // Decay time: 0–2 s
          float val = std::max(0.001f, (velocity / 127.0f) * 2.0f);
          for (auto &v : data->voices)
            v->adsr_decay = val;
          std::string msg = "dec: " + std::to_string(val) + "s";
          print_with_filename(data, msg);
        }
        if (key == 79) { // Sustain level: 0–1
          float val = velocity / 127.0f;
          for (auto &v : data->voices)
            v->adsr_sustain = val;
          std::string msg = "sus: " + std::to_string(val);
          print_with_filename(data, msg);
        }
        if (key == 72) { // Release time: 0–3 s
          float val = std::max(0.001f, (velocity / 127.0f) * 3.0f);
          for (auto &v : data->voices)
            v->adsr_release = val;
          std::string msg = "rel: " + std::to_string(val) + "s";
          print_with_filename(data, msg);
        }
        if (key == 48 && velocity > 0) { // next file
          int next = (data->current_file_index.load() + 1) % n;
          data->pending_file_index.store(next);
          data->file_change_pending.store(true);
        }
        if (key == 47 && velocity > 0) { // prev file
          int prev = (data->current_file_index.load() - 1 + n) % n;
          data->pending_file_index.store(prev);
          data->file_change_pending.store(true);
        }
        if (key == 80 && velocity > 0) { // CC80: toggle roboto
          data->roboto.store(!data->roboto.load());
          data->pending_file_index.store(data->current_file_index.load());
          print_with_filename(data, "Roboto fx = " +
                                        std::to_string(data->roboto.load()));
          data->file_change_pending.store(true);
        }
        if (key == 81 && velocity > 0) { // CC81: toggle whisper
          data->whisper.store(!data->whisper.load());
          data->pending_file_index.store(data->current_file_index.load());
          print_with_filename(data, "Whisper fx = " +
                                        std::to_string(data->roboto.load()));
          data->file_change_pending.store(true);
        }
        if (key == 82 && velocity > 0) { // CC82: toggle alien
          data->alien.store(!data->alien.load());
          data->pending_file_index.store(data->current_file_index.load());
          print_with_filename(data, std::string("Alien fx = ") +
                                        (data->alien.load() ? "on" : "off"));
          data->file_change_pending.store(true);
        }
      }
    }
  }

  MidiController() {
    try {
      midiIn = new RtMidiIn();
    } catch (RtMidiError &error) {
      std::cout << "MIDI in failed" << std::endl;
      error.printMessage();
    }

    unsigned int nPorts = midiIn->getPortCount();
    std::cout << "MIDI in: " << nPorts << std::endl;

    std::string portName;

    for (int i = 0; i < nPorts; i++) {
      try {
        portName = midiIn->getPortName(i);
      } catch (RtMidiError &error) {
        error.printMessage();
      }
      std::cout << "in" << i + 1 << ": " << portName << std::endl;
    }

    try {
      midiOut = new RtMidiOut();
    } catch (RtMidiError &error) {
      error.printMessage();
    }

    nPorts = midiOut->getPortCount();
    std::cout << "MIDI out: " << nPorts << std::endl;
    for (unsigned int i = 0; i < nPorts; i++) {
      try {
        portName = midiOut->getPortName(i);
        portNames.push_back(portName);
      } catch (RtMidiError &error) {
        error.printMessage();
      }
      std::cout << "out" << i + 1 << ": " << portName << std::endl;
    }
  }

  ~MidiController() {
    delete midiIn;
    delete midiOut;
  }
};
