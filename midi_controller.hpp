#pragma once
#include "RtMidi.h"
#include "sampler.cpp"


class MidiController
{
public:
  RtMidiIn *midiIn = 0;
  RtMidiOut *midiOut = 0;

  std::vector<std::string> portNames;

  static void callback(double deltatime, std::vector<unsigned char> *message, void *userData)
  {
    int key = (int)message->at(1);
    std::cout << key << " - ";
    int value = (int)message->at(0);
    std::cout << value << std::endl;

    callback_data_s *data = static_cast<callback_data_s*>(userData);
    if(value == 144 )
    {
        auto it = std::find_if(data->voices.begin(), data->voices.end(), [&](const std::shared_ptr<Sampler> s) {
            return !s->running && !s->stopping;
        });
        if(it != data->voices.end())
        {
          auto s = *it;
          if(s != nullptr)
          {
            s->index = 0;
            s->current_note = key;
            s->running = true;
          }
        }
    }
    if(value == 128 )
    {
        auto it = std::find_if(data->voices.begin(), data->voices.end(), [&, key](const std::shared_ptr<Sampler> s) {
            return s->running && !s->stopping && s->current_note == key;
        });
        if(it != data->voices.end())
        {
          auto s = *it;
          if(s != nullptr)
          {
            // s->index = 0;
            s->running = false;
            s->stopping = true;
          }
        }
    }
    // int value = (int)message->at(2);
    // std::cout << "Recieving [176, " << key << ", " << value <<"]" << std::endl;
    // Store *store = static_cast<Store *>(userData);
    // store->update(key, value);
  }

  MidiController()
  {
    try
    {
      midiIn = new RtMidiIn();
    }
    catch (RtMidiError &error)
    {
      std::cout << "Couldn't initialize RtMidiIn: " << std::endl;
      error.printMessage();
    }

    unsigned int nPorts = midiIn->getPortCount();
    std::cout << "\nThere are " << nPorts << " MIDI input sources available.\n";

    std::string portName;

    for (int i = 0; i < nPorts; i++)
    {
      try
      {
        portName = midiIn->getPortName(i);
      }
      catch (RtMidiError &error)
      {
        error.printMessage();
      }
      std::cout << "  Input Port #" << i + 1 << ": " << portName << '\n';
    }

    try
    {
      midiOut = new RtMidiOut();
    }
    catch (RtMidiError &error)
    {
      error.printMessage();
    }

    nPorts = midiOut->getPortCount();
    std::cout << "\nThere are " << nPorts << " MIDI output ports available.\n";
    for (unsigned int i = 0; i < nPorts; i++)
    {
      try
      {
        portName = midiOut->getPortName(i);
        portNames.push_back(portName);
      }
      catch (RtMidiError &error)
      {
        error.printMessage();
      }
      std::cout << "  Output Port #" << i + 1 << ": " << portName << '\n';
    }
    std::cout << '\n';
  }

  ~MidiController()
  {
    delete midiIn;
    delete midiOut;
  }
};