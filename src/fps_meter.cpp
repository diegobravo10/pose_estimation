#include "fps_meter.hpp"

FpsMeter::FpsMeter()
    : startTime_(std::chrono::steady_clock::now()),
      frameCount_(0),
      currentFps_(0.0) {
}

bool FpsMeter::update() {
    frameCount_++;

    const auto currentTime =
        std::chrono::steady_clock::now();

    const double elapsedSeconds =
        std::chrono::duration<double>(
            currentTime - startTime_
        ).count();

    if (elapsedSeconds >= 1.0) {
        currentFps_ =
            static_cast<double>(frameCount_) /
            elapsedSeconds;

        frameCount_ = 0;
        startTime_ = currentTime;

        return true;
    }

    return false;
}

double FpsMeter::fps() const {
    return currentFps_;
}

