# Cloudbuster

A command-line phase-vocoder sampler. It loads a folder of audio files, precomputes pitch-shifted samples across a MIDI note range using a phase-vocoder algorithm, and plays them back in real time via MIDI input. Time-stretching is independently controllable from pitch.

Supports 4-voice polyphony, real-time MIDI CC control, and several vocal/spectral effects.

# Building

- Clone with submodules: `git clone --recurse-submodules` or run `git submodule update --recursive --init` after cloning.
- On Linux, install ALSA dev kit: `sudo apt-get install libasound-dev`. See PortAudio docs for other platforms.
- Create a `build` dir, `cd` into it, and run `cmake ..`
- Run `make` to build.

# Usage

```
./cloudbuster -f <folder> [options] [effects]
```

`-f` points to a **folder** containing audio files (`.wav`, `.aif`, `.aiff`, `.flac`, `.ogg`). All files in the folder are loaded and can be cycled through via MIDI.

### Flags

| Flag | Description |
|------|-------------|
| `-f <path>` | Folder of audio files to load (default: `.`) |
| `-n <int>` | FFT size / window size (default: 1024) |
| `-h <int>` | Hop size divider — hop = window / this value (default: 8) |
| `-s <float>` | Time-stretch factor (default: 2.0) |
| `-fs <float>` | Internal sample rate (default: 8000) |
| `-min <int>` | Lowest MIDI note to precompute (default: 40) |
| `-max <int>` | Highest MIDI note to precompute (default: 90) |
| `-midi <int>` | MIDI port number to read from (default: 1) |
| `-device <int>` | PortAudio output device number (default: 0) |

### Effects (startup flags)

| Flag | Description |
|------|-------------|
| `roboto` | Enable robotization effect |
| `whisper` | Enable whisper effect |
| `alien` | Enable alien/ring-mod effect |
| `no-precompute` | Skip precomputation (samples are computed on demand) |

# MIDI CC Control

All parameters can be changed at runtime via MIDI Control Change messages.

| CC | Name | Range / Steps |
|----|------|---------------|
| 7 | Volume | 0–127 → 0.0–1.0 |
| 20 | N (FFT size) | 0–31→512, 32–63→1024, 64–95→2048, 96–127→4096 |
| 21 | Hop divisor | 0–31→2, 32–63→4, 64–95→8, 96–127→16 |
| 22 | Stretch | 0–127 → 0.25–4.0 |
| 47 | Previous file | trigger (velocity > 0) |
| 48 | Next file | trigger (velocity > 0) |
| 72 | Release time | 0–127 → 0–3 s |
| 73 | Attack time | 0–127 → 0–2 s |
| 75 | Decay time | 0–127 → 0–2 s |
| 79 | Sustain level | 0–127 → 0.0–1.0 |
| 80 | Roboto toggle | trigger (velocity > 0) |
| 81 | Whisper toggle | trigger (velocity > 0) |
| 82 | Alien toggle | trigger (velocity > 0) |

Changes to N, hop divisor, and stretch trigger a debounced voice rebuild.

# `midi.py` — Test Script

`midi.py` is a Python helper for sending MIDI messages to a running Cloudbuster instance from the terminal. It uses `amidi` (raw MIDI) and `aconnect` (ALSA sequencer) and requires a VirMIDI kernel module (`modprobe snd-virmidi`).

### Setup (one-time per session)

```sh
# Find available ports
./midi.py --list

# Connect VirMIDI → Cloudbuster (auto-detected)
./midi.py --connect
```

### Examples

```sh
./midi.py note 60                  # play middle C for 0.5s
./midi.py note 60 100 -d 2         # note 60, velocity 100, 2s duration
./midi.py cc 7 100                 # raw CC message
./midi.py volume 100               # set volume
./midi.py stretch 64               # set stretch (~2.1x)
./midi.py N 64                     # set N → 2048
./midi.py hop 96                   # set hop divisor → 16
./midi.py attack 64
./midi.py decay 32
./midi.py sustain 100
./midi.py release 80
./midi.py next                     # next file (CC48)
./midi.py prev                     # previous file (CC47)
./midi.py roboto                   # toggle roboto effect
./midi.py whisper                  # toggle whisper effect
./midi.py alien                    # toggle alien effect
```

### Options

| Option | Description |
|--------|-------------|
| `--port`, `-p` | amidi port (default: `hw:4,0`) |
| `--channel`, `-c` | MIDI channel 0–15 (default: 0) |
| `--list`, `-l` | List available MIDI ports and exit |
| `--connect` | Connect VirMIDI → Cloudbuster and exit |
