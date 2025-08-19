#include <cmath>

class BiquadLowpass {
public:
    BiquadLowpass(double sampleRate, double cutoffHz, double q = 0.707)
        : fs(sampleRate), cutoff(cutoffHz), Q(q)
    {
        updateCoefficients();
    }

    void setCutoff(double cutoffHz) {
        cutoff = cutoffHz;
        updateCoefficients();
    }

    void setQ(double q) {
        Q = q;
        updateCoefficients();
    }

    double process(double x) {
        // Direct Form I
        double y = b0 * x + b1 * x1 + b2 * x2
                         - a1 * y1 - a2 * y2;

        // shift states
        x2 = x1; x1 = x;
        y2 = y1; y1 = y;

        return y;
    }

    void reset() {
        x1 = x2 = y1 = y2 = 0.0;
    }

private:
    double fs;       // sample rate
    double cutoff;   // cutoff frequency (Hz)
    double Q;        // quality factor

    // coefficients
    double a1 = 0, a2 = 0, b0 = 0, b1 = 0, b2 = 0;

    // delay states
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;

    void updateCoefficients() {
        double omega = 2.0 * M_PI * cutoff / fs;
        double alpha = sin(omega) / (2.0 * Q);

        double cosw = cos(omega);
        double norm = 1.0 / (1.0 + alpha);

        b0 = (1.0 - cosw) * 0.5 * norm;
        b1 = (1.0 - cosw) * norm;
        b2 = b0;
        a1 = -2.0 * cosw * norm;
        a2 = (1.0 - alpha) * norm;
    }
};
