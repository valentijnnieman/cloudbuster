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

  // runtime parameter changes (require voice rebuild)
  std::atomic<int> pending_N{1024};
  std::atomic<int> pending_hop_size_div{8};
  std::atomic<float> pending_stretch{2.0f};
  std::atomic<bool> param_change_dirty{false};
  std::atomic<int64_t> param_last_change_ms{0};
} callback_data_s;

class MidiController {
public:
  RtMidiIn *midiIn = 0;
  RtMidiOut *midiOut = 0;

  std::vector<std::string> portNames;

  static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  static void print_with_filename(callback_data_s *data,
                                  const std::string &msg) {
    std::cout << std::filesystem::path(
                     data->file_list[data->current_file_index])
                     .filename()
                     .string()
              << std::endl;
    std::cout << msg << std::endl;
  }

  // A voice is available once it has never been assigned, or has finished its
  // envelope with no note-on still sitting in its queue. The queue check
  // matters: between the MIDI write and the audio thread's drain, `running` is
  // still false even though the voice is spoken for.
  static bool voice_is_free(const std::shared_ptr<Vocoder> &v) {
    return v->midi_note.load() == -1 ||
           (!v->running.load() &&
            PaUtil_GetRingBufferReadAvailable(&v->_note_queue) == 0);
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
      auto &voices = data->voices;
      if (voices.empty())
        return;

      // Retrigger a voice that is already sounding this note — including one
      // still in its release tail — instead of stealing a second voice for it.
      // Two voices on the same note is the doubling we fixed in the plugin.
      for (auto &v : voices) {
        if (v->midi_note.load() == key && !voice_is_free(v)) {
          v->stopping.store(false);
          NoteEvent ev{key};
          PaUtil_WriteRingBuffer(&v->_note_queue, &ev, 1);
          return;
        }
      }

      // Prefer a free voice; fall back to round-robin steal
      size_t idx = data->stln_voice;
      for (size_t i = 0; i < voices.size(); i++) {
        size_t candidate = (data->stln_voice + i) % voices.size();
        if (voice_is_free(voices[candidate])) {
          idx = candidate;
          break;
        }
      }
      data->stln_voice = (idx + 1) % voices.size();

      auto &s = voices[idx];
      s->midi_note.store(key);
      s->stopping.store(false);
      NoteEvent ev{key};
      PaUtil_WriteRingBuffer(&s->_note_queue, &ev, 1);
    }
    if (status == 0x80 || (status == 0x90 && velocity == 0)) { // Note Off
      // Match on midi_note, not current_note: a note-off arriving in the same
      // audio buffer as its note-on would find current_note still stale and
      // drop the release, leaving the voice sounding forever — and the next
      // press of that note doubling on top of it.
      for (auto &v : data->voices) {
        // voice_is_free skips voices whose midi_note is merely a leftover from
        // a note that already finished — matching one of those would swallow
        // the note-off and leave the voice that is actually sounding stuck on.
        if (v->midi_note.load() == key && !voice_is_free(v) &&
            !v->stopping.load()) {
          v->stopping.store(true); // triggers ADSR release
          // midi_note stays put: the voice owns the note through its release
          // tail, so a retrigger lands back on this same voice.
          break;
        }
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
      if (key == 20) { // CC20: N (512/1024/2048/4096)
        const int presets[] = {512, 1024, 2048, 4096};
        int i = std::min(3, (velocity * 4) / 128);
        data->pending_N.store(presets[i]);
        data->param_change_dirty.store(true);
        data->param_last_change_ms.store(now_ms());
        print_with_filename(data, "N=" + std::to_string(presets[i]));
      }
      if (key == 21) { // CC21: hop_size_div (2/4/8/16)
        const int presets[] = {2, 4, 8, 16};
        int i = std::min(3, (velocity * 4) / 128);
        data->pending_hop_size_div.store(presets[i]);
        data->param_change_dirty.store(true);
        data->param_last_change_ms.store(now_ms());
        print_with_filename(data, "hop/=" + std::to_string(presets[i]));
      }
      if (key == 22) { // CC22: stretch (0.25–4.0), applied live
        float val = 0.25f + (velocity / 127.0f) * 3.75f;
        for (auto &v : data->voices)
          v->stretch = val;
        data->pending_stretch.store(val);
        data->param_change_dirty.store(true);
        data->param_last_change_ms.store(now_ms());
        print_with_filename(data, "stretch=" + std::to_string(val));
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
          std::string msg = "loading...";
          print_with_filename(data, msg);
        }
        if (key == 47 && velocity > 0) { // prev file
          int prev = (data->current_file_index.load() - 1 + n) % n;
          data->pending_file_index.store(prev);
          data->file_change_pending.store(true);
          std::string msg = "loading...";
          print_with_filename(data, msg);
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
