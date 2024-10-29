# Cloudbuster
This is a command-line based program that takes an WAV file and calculates samples for a set of given midi notes, and plays them back using midi input. In other words, this is a command line sampler. It uses a phase vocoder algorithm, and can do pitch-shifting of samples independent of time. 

# Building
- install ALSA devkit: `sudo apt-get install libasound-dev`
- make `build` dir, cd into it, and run `cmake ..`
- run `make` to build program

# How to use
After building, run `./cloudbuster -f $FILE` where `$FILE` is a relative or absolute path to an audio WAV file. Other flags include `-midi $PORT` to change the midi port to read from, and `-device $DEVICE` to change the output device. Here's a list of all flags:

### flags
  - `-f`: WAV file to sample from
  - `-n`: the FFT size to use for the algorithm. Must be higher or even to window size.
  - `-w`: Size of the window (or frame) used by the STFT
  - `-h`: The hop size divider - hop size is calculated by dividing the window size by this value.
  - `-min`: The lowest midi note to be calculated. Any notes below this one won't have a sample.
  - `-max`: The highest midi note to be calculated. Any notes above this one won't have a sample.
  - `-midi`: The midi port number to use as input.
  - `-device`: The device number to use for audio output.
  - `no-unwrap`: Prevent unwrapping phase in the phase-vocoder algorithm. Can lead to interesting sounds. 
  - `instantaneous`: Calculate instantaneous frequency in pitch shifting algorithm. Can lead to interesting sounds.
  - `pitchshift`: Use the pitch-shifting algorithm. Samples are shifted in pitch but kept equal-length.
  - `phase-vocoder`: Use the basic phase-vocoder algorithm. Samples are shifted in pitch but are not kept the same length.

