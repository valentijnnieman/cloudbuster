#pragma once
#include <memory>

#include <samplerate.h>
#include "hanning.hpp"
#include "sndfile.hh"
#include "pffft.hpp"
#include "fftw3.h"
#include <map>
#include <matplot/matplot.h>

using namespace matplot;

class Sampler;

typedef struct
{
    SNDFILE     *file;
    SF_INFO      info;
    std::vector<std::shared_ptr<Sampler> > voices;
    std::vector<int> indices;
    // std::vector<float> samples;
    std::vector<int> notes;
    int max;
    int index;
} callback_data_s;

class Sampler
{
private:
    SNDFILE *file;
    SF_INFO info;

    int sig_len, analysis_hopsize, synthesis_hopsize;
    int N;
    int window_size;
    int hop_size_div;
    float pitch_ratio;

    float* in;
    fftwf_complex *out;

    fftwf_plan p;
    fftwf_plan pi;

    // typedef pffft::Fft<float> FFT_T;
    // pffft::Fft<float> fft;

public:
    int index = 0;
    int current_note = 0;
    bool running = false;
    bool stopping = false;
    int it;

    bool FANTASY;
    bool GONGO;
    bool INSTANTANEOUS;

    int fs;
    std::vector<float> samples;
    std::map<int, std::vector<float> > key_samples;

    int frames_size()
    {
        return key_samples[current_note].size();
    }

    void print_stats()
    {
        std::cout << "samples size: " << samples.size();
        std::cout << std::endl;
        // std::cout << "window[0]: " << window[0] << std::endl;
    }
    Sampler(const std::string& filename, int N = 1024, int window_size = 1024, int hop_size_div = 4)
    : it(it), N(N), window_size(window_size), hop_size_div(hop_size_div)
    {
        /* Open the soundfile */
        file = sf_open(filename.c_str(), SFM_READ, &info);
        samples.resize(info.frames * info.channels);

        sf_readf_float(file, &samples[0], info.frames);

        fs = info.samplerate;
        sig_len = samples.size();

        // in = (float*) fftwf_malloc(sizeof(float) * window_size);
        // out = (float*) fftwf_malloc(sizeof(float) * window_size);
        // fft_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * window_size);
        // fft_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * window_size);


        pitch_ratio = 1.0f;
        /* Close the soundfile */
        sf_close(file);
    }
    ~Sampler()
    {
        free(file);
        fftwf_destroy_plan(p);
        fftwf_free(in); 
        fftwf_free(out);
    }

    std::vector<float> get_samples(int frameCount, float amp)
    {
        std::vector<float> out(frameCount, 0.0f);
        if(current_note != 0)
        {
            int left = (key_samples[current_note].size()-1) - index;

            if(left <= frameCount)
            {
                // final buffer
                for(int i = 0; i < left; i++)
                {
                    out[i] += key_samples[current_note][index + i] * amp;
                }
                // index += left;
                this->running = false;
                this->stopping = false;
            }
            else
            {
                for(int i = 0; i < frameCount; i++)
                {
                    out[i] += (key_samples[current_note][index + i] * amp);
                }
                index += frameCount;
            }
        }

        return out;
    }

    float note_to_freq(int note)
    {
        float a = 440.0f; //frequency of A (conmon value is 440Hz)
        // float d = 587.33; // freq of D note
        // return a * 2^((note−69)/12);

        return a * pow(2.0f, ((note - 69.0f) / 12.0f));
    }

    void forward_fft(float* time_data, std::complex<float>* freq_data)
    {
        out = reinterpret_cast<fftwf_complex*>(freq_data);
        in = time_data;
        p = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
        fftwf_execute(p);
    }

    void ifft(float* time_data, std::complex<float>* freq_data)
    {
        out = reinterpret_cast<fftwf_complex*>(freq_data);
        in = time_data;
        pi = fftwf_plan_dft_c2r_1d(N, out, in, FFTW_ESTIMATE);
        fftwf_execute(pi);
    }

    void resample(float* data_in, float* data_out, int input_size, int output_size, float pitch_ratio)
    {

        SRC_DATA src_data;
        src_data.data_in = data_in;
        src_data.data_out = data_out;
        src_data.input_frames = input_size;
        src_data.output_frames = output_size;
        src_data.src_ratio = pitch_ratio;

        src_simple(&src_data, 4, 2);
    }


    // Normalize to [0,2PI):
    // float normalize_phase(float x)
    // {
    //     x = fmod(x, 2*PI);
    //     if (x < 0)
    //         x += 2*PI;
    //     return x;
    // };

    // // unwrap phase [-PI,PI]
    // float unwrap(float previous_angle, float new_angle) {
    //     float d = new_angle - previous_angle;
    //     d = d > M_PI ? d - 2 * M_PI : (d < -M_PI ? d + 2 * M_PI : d);
    //     return previous_angle + d;
    // }

    // calculate sample with phase vocoder
    void calculate_sample_pv(callback_data_s& data, int note)
    {
        data.index = 0;
        data.max = samples.size()* 2;

        float freq = note_to_freq(note);
        float div = 440.0;
        float r = div / freq;
        pitch_ratio = r;

        std::cout << "calculating sample for note: " << note << std::endl;
        std::vector<float> out_samples(samples.size() / pitch_ratio, 0.0f);
        std::vector<float> resampled(samples.size(), 0.0f);

        std::cout << "samples size: " << samples.size() << std::endl;
        std::cout << "out_samples size: " << out_samples.size() << std::endl;
        std::cout << "resampled size: " << resampled.size() << std::endl;

        std::vector<float> window = hanning(window_size, 0);

        synthesis_hopsize = window_size/hop_size_div;
        analysis_hopsize = synthesis_hopsize * pitch_ratio;

        int read_ptr = 0;
        int write_ptr = 0;

        while(read_ptr <= samples.size() - read_ptr)
        {
            std::vector<float> buffer(window_size, 0.0f);
            std::vector<float> out(window_size, 0.0f);
            std::vector<float> phi(window_size, 0.0f);

            std::vector<std::complex<float> > resynth_buffer(window_size/2+1, 0.0f);
            std::vector<std::complex<float> > fft_buffer(window_size/2+1, 0.0f);

            // read M (window size) samples into buffer
            for(int i = 0; i < window_size; i++)
            {
                // multiply samples by analysis window w[m] of length window_size
                buffer[i] = window[i] * samples[read_ptr + i];
            }

            forward_fft((float*)&buffer[0], (std::complex<float>*)&fft_buffer[0]);
            for(int i = 0; i < window_size/2+1; i++)
            {
                // the instantaneous phase or local phase (or simply, phase!)
                // is calculated as arg(c(t)) where c(t) is a complex number and 
                // t is a time increment (like "i" in a for loop).
                float phase = std::arg(fft_buffer[i]);
                float previous_phase = 0.0f;
                if(i>0)
                {
                    previous_phase = std::arg(fft_buffer[i-1]);
                }

                // the instantaneous frequency is the "temporal rate of change" of the
                // instantaneous phase. When the phase is constrained to the interval of 
                // (-pi, pi) or (0, 2pi), which is the principal value (?), it is called
                // the "wrapped phase". Otherwise, it is called the "unwrapped phase".

                float amplitude = abs(fft_buffer[i]);

                // float freq = 2.0 * M_PI * (float)i / N;
                float freq = i * fs / N;
                float target = previous_phase + (freq*analysis_hopsize);
                float deviation = phase - target;
                float increment = (freq*analysis_hopsize) + deviation;

                // fi is the instantaneous frequency: a more accurate frequency measurement
                if(INSTANTANEOUS)
                {
                    float fi = increment / (2*PI*analysis_hopsize) * fs;
                    freq = fi;
                }
                float delta_phi = (freq*analysis_hopsize) + std::arg(phase - previous_phase - (freq*analysis_hopsize));
                if(GONGO)
                {
                    delta_phi = (freq*analysis_hopsize) + phase - previous_phase - (freq*analysis_hopsize);
                }

                if(FANTASY)
                {
                    phi[i] = delta_phi * synthesis_hopsize;
                }
                else
                {
                    phi[i] = std::arg(delta_phi * synthesis_hopsize);
                }

                // phi[i] = previous_phase + increment;

                resynth_buffer[i].real(amplitude*cos(phi[i]));
                resynth_buffer[i].imag(amplitude*sin(phi[i]));
            }

            ifft((float*)&out[0], (std::complex<float>*)&resynth_buffer[0]);
            // fft.inverse((std::complex<float>*)&resynth_buffer[0], (float*)&out[0]);

            for(int i = 0; i < window_size; i++)
            {
                out_samples[write_ptr + i] += window[i] * (out[i]);
            }

            read_ptr += analysis_hopsize;
            write_ptr += synthesis_hopsize;
        }

        resample((float*)&out_samples[0], (float*)&resampled[0], out_samples.size()/2, resampled.size()/2, pitch_ratio);

        key_samples.insert(std::pair<int, std::vector<float> >(note, resampled));
    }

    void calculate_sample(callback_data_s& data, int note)
    {
        data.index = 0;
        data.max = samples.size();
        std::cout << "calculating sample for note: " << note << std::endl;
        std::vector<float> out_samples(samples.size() / pitch_ratio, 0.0f);
        std::vector<float> resampled(samples.size(), 0.0f);

        float freq = note_to_freq(note);
        float div = 440.0f;
        float r = div / freq;
        pitch_ratio = r;

        resampled.resize(samples.size() * pitch_ratio, 0.0f);
        // std::fill(resampled.begin(), resampled.end(), 0.0f);

        out_samples = samples;

        resample((float*)&out_samples[0], (float*)&resampled[0],out_samples.size()/2, resampled.size()/2, pitch_ratio);

        key_samples.insert(std::pair<int, std::vector<float> >(note, resampled));
    }
};
