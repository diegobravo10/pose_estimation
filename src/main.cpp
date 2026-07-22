#include "camera.hpp"
#include "csv_logger.hpp"
#include "fps_meter.hpp"
#include "pose_estimator.hpp"
#include "yolo_pose.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

int main(
    int argc,
    char* argv[]
) {

    std::string modelType =
        "movenet";

    std::string source = "4";

    bool usingCamera = true;

    int requestedWidth = 640;

    int requestedHeight = 480;

    try {

        if (argc >= 2) {
            modelType = argv[1];
        }

        if (argc >= 3) {
            source = argv[2];
        }

        if (argc >= 4) {
            requestedWidth =
                std::stoi(argv[3]);
        }

        if (argc >= 5) {
            requestedHeight =
                std::stoi(argv[4]);
        }

    } catch (const std::exception& error) {

        std::cerr
            << "Error en argumentos: "
            << error.what()
            << "\n";

        std::cerr
            << "Uso con webcam:\n"
            << "./tesis_pose "
            << "[movenet|yolo] "
            << "[camara] "
            << "[ancho] "
            << "[alto]\n\n";

        std::cerr
            << "Uso con video:\n"
            << "./tesis_pose "
            << "[movenet|yolo] "
            << "[video.mp4]\n";

        return 1;
    }

    if (
        modelType != "movenet" &&
        modelType != "yolo"
    ) {

        std::cerr
            << "Modelo no valido: "
            << modelType
            << "\n";

        std::cerr
            << "Usa movenet o yolo.\n";

        return 1;
    }

    Camera camera;

    std::unique_ptr<
        PoseEstimator
    > poseEstimator;

    std::unique_ptr<
        YoloPose
    > yoloPose;

    try {

        if (
            modelType ==
            "movenet"
        ) {

            poseEstimator =
                std::make_unique<
                    PoseEstimator
                >();

        } else {

            yoloPose =
                std::make_unique<
                    YoloPose
                >(
                    "model/yolov8n-pose.onnx"
                );
        }

    } catch (
        const std::exception& error
    ) {

        std::cerr
            << "Error cargando modelo: "
            << error.what()
            << "\n";

        return 1;
    }

    bool sourceOpened = false;

    try {

        std::size_t position = 0;

        const int cameraIndex =
            std::stoi(
                source,
                &position
            );

        /*
        * Si todo el texto se pudo convertir
        * a numero, asumimos que es una webcam.
        */
        if (
            position ==
            source.size()
        ) {

            usingCamera = true;

            sourceOpened =
                camera.open(
                    cameraIndex,
                    requestedWidth,
                    requestedHeight
                );

        } else {

            usingCamera = false;

            sourceOpened =
                camera.openVideo(
                    source
                );
        }

    } catch (
        const std::exception&
    ) {

        /*
        * Si no se puede convertir a numero,
        * asumimos que es una ruta de video.
        */

        usingCamera = false;

        sourceOpened =
            camera.openVideo(
                source
            );
    }

    if (!sourceOpened) {
        return 1;
    }

    const int realWidth =
        camera.width();

    const int realHeight =
        camera.height();

    FpsMeter fpsMeter;

    const std::string csvPath =
        modelType == "yolo"
            ? "results/yolo_pose.csv"
            : "results/movenet_pose.csv";

    CsvLogger logger(
        csvPath
    );

    cv::Mat frame;

    double inferenceMilliseconds =
        0.0;

    std::cout
        << "\n============================\n"
        << "Modelo: "
        << modelType
        << "\n"
        << "Fuente: "
        << source
        << "\n"
        << "Resolucion: "
        << realWidth
        << "x"
        << realHeight
        << "\n"
        << "============================\n";

    std::cout
        << "\nControles:\n"
        << "  q   salir\n"
        << "  ESC salir\n\n";

    while (
        true
    ) {

        if (!camera.read(frame)) {

            if (usingCamera) {

                std::cerr
                    << "Error: no se pudo leer "
                    << "un frame de la camara.\n";

            } else {

                std::cout
                    << "\nFin del video.\n";
            }

            break;
        }

        try {

            const auto inferenceStart =
                std::chrono::
                    steady_clock::
                    now();

            if (
                modelType ==
                "movenet"
            ) {

                const auto keypoints =
                    poseEstimator
                        ->estimate(
                            frame
                        );

                const auto inferenceEnd =
                    std::chrono::
                        steady_clock::
                        now();

                inferenceMilliseconds =
                    std::chrono::
                        duration<
                            double,
                            std::milli
                        >(
                            inferenceEnd -
                            inferenceStart
                        )
                        .count();

                poseEstimator
                    ->drawPose(
                        frame,
                        keypoints
                    );

            } else {

                const auto detections =
                    yoloPose
                        ->estimate(
                            frame
                        );

                const auto inferenceEnd =
                    std::chrono::
                        steady_clock::
                        now();

                inferenceMilliseconds =
                    std::chrono::
                        duration<
                            double,
                            std::milli
                        >(
                            inferenceEnd -
                            inferenceStart
                        )
                        .count();

                yoloPose
                    ->drawPoses(
                        frame,
                        detections
                    );
            }

        } catch (
            const std::exception& error
        ) {

            std::cerr
                << "\nError: "
                << error.what()
                << "\n";

            break;
        }

        const bool fpsUpdated =
            fpsMeter.update();

        if (
            fpsUpdated
        ) {

            std::cout
                << "\rModelo: "
                << modelType
                << " | FPS: "
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
            cv::Point(
                20,
                35
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(
                0,
                255,
                0
            ),
            2,
            cv::LINE_AA
        );

        std::ostringstream
            inferenceText;

        inferenceText
            << "Inferencia: "
            << std::fixed
            << std::setprecision(1)
            << inferenceMilliseconds
            << " ms";

        cv::putText(
            frame,
            inferenceText.str(),
            cv::Point(
                20,
                70
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(
                0,
                255,
                0
            ),
            2,
            cv::LINE_AA
        );

        const std::string modelText =
            "Modelo: " +
            modelType;

        cv::putText(
            frame,
            modelText,
            cv::Point(
                20,
                105
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(
                0,
                255,
                255
            ),
            2,
            cv::LINE_AA
        );

        cv::imshow(
            "Sistema de estimacion de pose",
            frame
        );

        const int key =
            cv::waitKey(1);

        if (
            key == 27 ||
            key == 'q'
        ) {
            break;
        }
    }

    camera.release();

    cv::destroyAllWindows();

    std::cout
        << "\nPrograma finalizado.\n";

    std::cout
        << "Resultados: "
        << csvPath
        << "\n";

    return 0;
}