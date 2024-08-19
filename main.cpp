#include <stdio.h>
#include <string.h>
#include <complex>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <vector>

#include "portaudio.h"

#include "midi_controller.hpp"
#include <numbers>
#include <cmath>

// #include <matplot/matplot.h>
// using namespace matplot;

// Normalize to [0,2PI):
double phaseNorm(float x)
{
    x = fmod(x, 2*PI);
    if (x < 0)
        x += 2*PI;
    return x;
};

void cartesianToPolar(float x, float y, float &r, float &q ) {
    r = sqrt(x * x + y * y); q = atan(y / x);
}

// unwrap phase [-PI,PI]
std::vector<float> phaseUnwrap(std::vector<float> in)
{
    // Normalize to [0,2PI):
    std::transform(in.begin(),in.end(),in.begin(),[&](float d){ return phaseNorm(d); });

    // unwrap iteration
    for(size_t i = 0; i < in.size()-1; ++i)
    {
        int n2PiJump = in[i] / (2*PI);
        in[i+1] += n2PiJump * 2*PI;
        if(in[i]-in[i+1] > PI)
        {
            in[i+1] += PI;
        }
    }
    return in;
}


#define FRAMES_PER_BUFFER   (512)


static
int
callback
    (const void                     *input
    ,void                           *output
    ,unsigned long                   frameCount
    ,const PaStreamCallbackTimeInfo *timeInfo
    ,PaStreamCallbackFlags           statusFlags
    ,void                           *userData
    )
{
    callback_data_s *data = (callback_data_s*)userData;
    float *out = (float*)output;
    memset(out, 0, sizeof(float) * frameCount * 2);

    std::vector<float> buffer(frameCount * 2, 0.0f);
    float amp = 1.0f / (float)data->voices.size();

    for(auto sampler : data->voices)
    {
        if(sampler->running && !sampler->stopping)
        {
            std::vector<float> samples = sampler->get_samples(frameCount * 2, amp);

            for(int i = 0; i < frameCount * 2; i++)
            {
                buffer[i] += samples[i];
            }
        }
        if(sampler->stopping && !sampler->running)
        {
            std::vector<float> samples = sampler->get_samples(frameCount * 2, amp);

            if(sampler->index >= sampler->frames_size()/2)
            {
                sampler->stopping = false;
                sampler->running = false;
            }
            else
            {
                for(int i = 0; i < frameCount * 2; i++)
                {
                    buffer[i] += samples[i];
                }
            }
        }
    }
    for(int i = 0; i < frameCount * 2; i++)
    {
        *out++ = buffer[i];
    }
    
    return paContinue;
}



int main(int argc, const char * argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);

    bool fantasy = false;
    bool gongo = false;
    int N = 1024;
    int window_size = 1024;
    int min_note = 20;
    int max_note = 90;
    int hop_size_div = 8;
    int midi_in = 1;
    int midi_out = 1;
    std::string filename = "/home/valentijn/dev/c++/cloudbuster/fairchild/samples/Piano_note_a1.wav";

    for(int i = 0; i < args.size(); i++)
    {
        if(args[i] == "-f")
        {
            filename = args[i+1];
            std::cout << "[Sampler] using file: " << filename << std::endl;
        }
        if(args[i] == "-n")
        {
            N = stoi(args[i+1]);
            std::cout << "[Sampler] N (fft) size: " << N << std::endl;
        }
        if(args[i] == "-w")
        {
            window_size = stoi(args[i+1]);
            std::cout << "[Sampler] window size: " << window_size << std::endl;
        }
        if(args[i] == "-h")
        {
            hop_size_div = stoi(args[i+1]);
            std::cout << "[Sampler] hop size divide by: " << hop_size_div << std::endl;
        }
        if(args[i] == "-min")
        {
            min_note = stoi(args[i+1]);
            std::cout << "[Sampler] minimum (midi) note: " << min_note << std::endl;
        }
        if(args[i] == "-max")
        {
            max_note = stoi(args[i+1]);
            std::cout << "[Sampler] maximum (midi) note: " << max_note << std::endl;
        }
        if(args[i] == "-midi")
        {
            midi_in = stoi(args[i+1]);
            midi_out = midi_in;
            std::cout << "[Sampler] midi i/o ports: " << midi_in<< std::endl;
        }
        else
        {
            if(args[i] == "fantasy")
            {
                fantasy = true;
                std::cout << "[Sampler] using FANTASY preset" << std::endl;
            }
            if(args[i] == "gongo")
            {
                gongo = true;
                std::cout << "[Sampler] using GONGO preset" << std::endl;
            }
        }
    }

    // SndfileHandle file;
    Sampler sampler = Sampler(filename, N, window_size, hop_size_div);
    sampler.FANTASY = fantasy;
    sampler.GONGO = gongo;

    PaStream *stream;
    PaError error;
    callback_data_s data;

    MidiController ctrl;

    for(auto& name : ctrl.portNames)
    {
        std::cout << name << std::endl;
    }

    ctrl.midiIn->openPort(midi_in);
    ctrl.midiIn->setCallback(ctrl.callback, &data);

    ctrl.midiOut->openPort(midi_out);

    std::cout << "calculating samples..." <<std::endl;
    for(int i = min_note; i < max_note; i++)
    {
        sampler.calculate_sample_pv(data, i);
    }

    // data.voices.push_back(std::make_shared<Sampler>(sampler));
    std::cout << "copying samplers for voices..." <<std::endl;

    for(int j = 0; j < 32; j++)
    {
        std::cout << "creating sampler #: " << j << std::endl;
        Sampler *s = new Sampler(sampler);
        s->it = j;

        s->print_stats();
        data.voices.push_back(std::shared_ptr<Sampler>(s));
    }
    std::cout << "Done! opening stream..." <<std::endl;
    // data.samples = out_samples;

    // data viz section!

    // std::vector<double> x = linspace(0, out_samples.size());
    // plot(x, samples);
    // hold(on);
    // plot(x, out_samples);
    // show();


    /* init portaudio */
    error = Pa_Initialize();
    if(error != paNoError)
    {
        fprintf(stderr, "Problem initializing\n");
        return 1;
    }
    
    /* Open PaStream with values read from the file */
    error = Pa_OpenDefaultStream(&stream
                                 ,0                     /* no input */
                                //  ,data.info.channels         /* stereo out */
                                ,2
                                 ,paFloat32             /* floating point */
                                //  ,file.samplerate()
                                ,sampler.fs
                                // ,44100
                                 ,512
                                 ,callback
                                 ,&data);        /* our sndfile data struct */
    if(error != paNoError)
    {
        fprintf(stderr, "Problem opening Default Stream\n");
        return 1;
    }
    
    /* Start the stream */
    error = Pa_StartStream(stream);
    if(error != paNoError)
    {
        fprintf(stderr, "Problem opening starting Stream\n");
        return 1;
    }

    /* Run until EOF is reached */
    while(Pa_IsStreamActive(stream))
    {
        Pa_Sleep(1000);
    }

    /*  Shut down portaudio */
    error = Pa_CloseStream(stream);
    if(error != paNoError)
    {
        fprintf(stderr, "Problem closing stream\n");
        return 1;
    }
    
    error = Pa_Terminate();
    if(error != paNoError)
    {
        fprintf(stderr, "Problem terminating\n");
        return 1;
    }
    
    return 0;
}
