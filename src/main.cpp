#include "camera.hpp"
#include "csv_logger.hpp"
#include "fps_meter.hpp"
#include "pose_estimator.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    int cameraIndex = 0;
    int requestedWidth = 640;
    int requestedHeight = 480;

    try {
        if (argc >= 2) {
            cameraIndex = std::stoi(argv[1]);
        }

        if (argc >= 3) {
            requestedWidth = std::stoi(argv[2]);
        }

        if (argc >= 4) {
            requestedHeight = std::stoi(argv[3]);
        }
    } catch (const std::exception& error) {
        std::cerr
            << "Error en los argumentos: "
            << error.what() << "\n";

        std::cerr
            << "Uso: ./tesis_pose "
            << "[camara] [ancho] [alto]\n";

        return 1;
    }

    Camera camera;

    std::unique_ptr<PoseEstimator> poseEstimator;
    try {
        poseEstimator = std::make_unique<PoseEstimator>();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    if (!camera.open(
            cameraIndex,
            requestedWidth,
            requestedHeight
        )) {
        return 1;
    }

    const int realWidth = camera.width();
    const int realHeight = camera.height();

    FpsMeter fpsMeter;

    CsvLogger logger(
        "results/camera_baseline.csv"
    );

    if (!logger.isOpen()) {
        std::cerr
            << "Advertencia: no se pudo abrir "
            << "el archivo CSV.\n";
    }

    cv::Mat frame;
    double inferenceMilliseconds = 0.0;

    std::cout << "\nControles:\n";
    std::cout << "  q   salir\n";
    std::cout << "  ESC salir\n\n";

    while (true) {
        if (!camera.read(frame)) {
            std::cerr
                << "Error: no se pudo leer "
                << "un frame de la cámara.\n";
            break;
        }

        try {
            const auto inferenceStart = std::chrono::steady_clock::now();
            const std::vector<Keypoint> keypoints = poseEstimator->estimate(frame);
            const auto inferenceEnd = std::chrono::steady_clock::now();

            inferenceMilliseconds =
                std::chrono::duration<double, std::milli>(
                    inferenceEnd - inferenceStart
                ).count();

            poseEstimator->drawPose(frame, keypoints);
        } catch (const std::exception& error) {
            std::cerr << "\nError: " << error.what() << "\n";
            break;
        }

        const bool fpsUpdated = fpsMeter.update();

        if (fpsUpdated) {
            std::cout
                << "\rFPS: "
                << std::fixed
                << std::setprecision(2)
                << fpsMeter.fps()
                << " | Inferencia: "
                << inferenceMilliseconds
                << " ms"
                << std::flush;

            logger.log(
                fpsMeter.fps(),
                realWidth,
                realHeight
            );
        }

        std::ostringstream fpsText;

        fpsText
            << "FPS: "
            << std::fixed
            << std::setprecision(1)
            << fpsMeter.fps();

        cv::putText(
            frame,
            fpsText.str(),
            cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA
        );

        std::ostringstream inferenceText;
        inferenceText << "Inferencia: " << std::fixed << std::setprecision(1)
                      << inferenceMilliseconds << " ms";

        cv::putText(
            frame,
            inferenceText.str(),
            cv::Point(20, 70),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA
        );

        const std::string resolutionText =
            "Resolucion: " + std::to_string(realWidth) + "x" +
            std::to_string(realHeight);

        cv::putText(
            frame,
            resolutionText,
            cv::Point(20, 105),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA
        );

        cv::imshow(
            "Sistema de estimacion de pose",
            frame
        );

        const int key = cv::waitKey(1);

        if (key == 27 || key == 'q') {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    std::cout << "\nPrograma finalizado.\n";
    std::cout
        << "Resultados guardados en: "
        << "results/camera_baseline.csv\n";

    return 0;
}
