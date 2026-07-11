#ifndef FPS_METER_HPP
#define FPS_METER_HPP

#include <chrono>

class FpsMeter {
public:
    FpsMeter();

    /*
     * Se llama una vez por frame.
     * Devuelve true cuando se actualiza el FPS.
     */
    bool update();

    double fps() const;

private:
    std::chrono::steady_clock::time_point startTime_;
    int frameCount_;
    double currentFps_;
};

#endif
