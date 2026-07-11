#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera() = default;

    bool open(int cameraIndex, int width, int height);
    bool read(cv::Mat& frame);
    void release();

    bool isOpened() const;

    int width() const;
    int height() const;

private:
    cv::VideoCapture capture_;
};

#endif

