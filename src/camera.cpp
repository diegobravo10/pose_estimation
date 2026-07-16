#include "camera.hpp"

#include <iostream>

bool Camera::open(
    int cameraIndex,
    int width,
    int height
) {
    capture_.open(
        cameraIndex,
        cv::CAP_V4L2
    );

    if (!capture_.isOpened()) {
        std::cerr
            << "Error: no se pudo abrir la camara "
            << cameraIndex
            << ".\n";

        return false;
    }

    capture_.set(
        cv::CAP_PROP_FRAME_WIDTH,
        width
    );

    capture_.set(
        cv::CAP_PROP_FRAME_HEIGHT,
        height
    );

    /*
     * Reduce el retraso causado por frames
     * almacenados por la webcam.
     */
    capture_.set(
        cv::CAP_PROP_BUFFERSIZE,
        1
    );

    std::cout
        << "Camara abierta correctamente.\n";

    std::cout
        << "Resolucion real: "
        << this->width()
        << "x"
        << this->height()
        << "\n";

    return true;
}


bool Camera::openVideo(
    const std::string& videoPath
) {
    capture_.open(videoPath);

    if (!capture_.isOpened()) {
        std::cerr
            << "Error: no se pudo abrir el video: "
            << videoPath
            << "\n";

        return false;
    }

    std::cout
        << "Video abierto correctamente.\n";

    std::cout
        << "Archivo: "
        << videoPath
        << "\n";

    std::cout
        << "Resolucion: "
        << width()
        << "x"
        << height()
        << "\n";

    std::cout
        << "FPS original del video: "
        << fps()
        << "\n";

    std::cout
        << "Numero de frames: "
        << frameCount()
        << "\n";

    return true;
}


bool Camera::read(
    cv::Mat& frame
) {
    return capture_.read(frame);
}


void Camera::release() {
    if (capture_.isOpened()) {
        capture_.release();
    }
}


bool Camera::isOpened() const {
    return capture_.isOpened();
}


int Camera::width() const {
    return static_cast<int>(
        capture_.get(
            cv::CAP_PROP_FRAME_WIDTH
        )
    );
}


int Camera::height() const {
    return static_cast<int>(
        capture_.get(
            cv::CAP_PROP_FRAME_HEIGHT
        )
    );
}


double Camera::fps() const {
    return capture_.get(
        cv::CAP_PROP_FPS
    );
}


double Camera::frameCount() const {
    return capture_.get(
        cv::CAP_PROP_FRAME_COUNT
    );
}