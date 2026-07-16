#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/opencv.hpp>

#include <string>

class Camera {
public:
    Camera() = default;

    // Webcam
    bool open(int cameraIndex, int width, int height);

    // Archivo de video
    bool openVideo(const std::string& videoPath);

    bool read(cv::Mat& frame);
    void release();

    bool isOpened() const;

    int width() const;
    int height() const;

    double fps() const;
    double frameCount() const;

private:
    cv::VideoCapture capture_;
};

#endif