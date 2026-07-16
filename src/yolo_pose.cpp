#include "yolo_pose.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

std::string resolveYoloModelPath(
    const std::string& requestedPath
) {
    if (std::filesystem::exists(requestedPath)) {
        return requestedPath;
    }

    throw std::runtime_error(
        "modelo YOLO Pose no encontrado: " +
        requestedPath
    );
}

Ort::SessionOptions makeYoloSessionOptions() {
    Ort::SessionOptions options;

    // Tu Toshiba es CPU, empezamos con 2 hilos.
    // Después podemos experimentar con 1, 2 y 4.
    options.SetIntraOpNumThreads(2);

    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    return options;
}

Ort::Session makeYoloSession(
    const Ort::Env& env,
    const std::string& modelPath,
    const Ort::SessionOptions& options
) {
    try {
        return Ort::Session(
            env,
            modelPath.c_str(),
            options
        );
    } catch (const Ort::Exception& error) {
        throw std::runtime_error(
            std::string(
                "error al cargar YOLO Pose: "
            ) +
            error.what()
        );
    }
}

float intersectionOverUnion(
    const cv::Rect& a,
    const cv::Rect& b
) {
    const int intersectionX1 =
        std::max(a.x, b.x);

    const int intersectionY1 =
        std::max(a.y, b.y);

    const int intersectionX2 =
        std::min(
            a.x + a.width,
            b.x + b.width
        );

    const int intersectionY2 =
        std::min(
            a.y + a.height,
            b.y + b.height
        );

    const int intersectionWidth =
        std::max(
            0,
            intersectionX2 - intersectionX1
        );

    const int intersectionHeight =
        std::max(
            0,
            intersectionY2 - intersectionY1
        );

    const float intersectionArea =
        static_cast<float>(
            intersectionWidth *
            intersectionHeight
        );

    const float unionArea =
        static_cast<float>(
            a.area() +
            b.area()
        ) -
        intersectionArea;

    if (unionArea <= 0.0F) {
        return 0.0F;
    }

    return intersectionArea / unionArea;
}

}  // namespace


YoloPose::YoloPose(
    const std::string& modelPath
)
    : modelPath_(
          resolveYoloModelPath(modelPath)
      ),
      env_(
          ORT_LOGGING_LEVEL_WARNING,
          "tesis_pose_yolo"
      ),
      sessionOptions_(
          makeYoloSessionOptions()
      ),
      session_(
          makeYoloSession(
              env_,
              modelPath_,
              sessionOptions_
          )
      ) {

    if (
        session_.GetInputCount() != 1 ||
        session_.GetOutputCount() < 1
    ) {
        throw std::runtime_error(
            "YOLO Pose debe tener una entrada "
            "y al menos una salida"
        );
    }

    Ort::AllocatorWithDefaultOptions allocator;

    const auto inputName =
        session_.GetInputNameAllocated(
            0,
            allocator
        );

    const auto outputName =
        session_.GetOutputNameAllocated(
            0,
            allocator
        );

    inputName_ = inputName.get();
    outputName_ = outputName.get();

    const auto inputTypeInfo =
    session_.GetInputTypeInfo(0);

    const auto outputTypeInfo =
        session_.GetOutputTypeInfo(0);

    const auto inputInfo =
        inputTypeInfo.GetTensorTypeAndShapeInfo();

    const auto outputInfo =
        outputTypeInfo.GetTensorTypeAndShapeInfo();

    inputShape_ =
        inputInfo.GetShape();

    const auto outputShape =
        outputInfo.GetShape();

    std::cout << "Input shape detectado por C++: [";

    for (std::size_t i = 0; i < inputShape_.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << inputShape_[i];
    }

    std::cout << "]\n";

    std::cout << "Output shape detectado por C++: [";

    for (std::size_t i = 0; i < outputShape.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << outputShape[i];
    }

    std::cout << "]\n";

    if (
        inputShape_.size() != 4 ||
        inputShape_[0] != 1 ||
        inputShape_[1] != 3 ||
        inputShape_[2] != kInputHeight ||
        inputShape_[3] != kInputWidth
    ) {
        throw std::runtime_error(
            "entrada inesperada para YOLO Pose"
        );
    }

    if (
        outputShape.size() != 3 ||
        outputShape[0] != 1 ||
        outputShape[1] != kFeatureCount ||
        outputShape[2] != kCandidateCount
    ) {
        throw std::runtime_error(
            "salida inesperada para YOLO Pose"
        );
    }

    std::cout
        << "\nModelo YOLO Pose cargado correctamente.\n"
        << "Runtime: ONNX Runtime\n"
        << "Execution Provider: CPU\n"
        << "Input: [1,3,640,640]\n"
        << "Output: [1,56,8400]\n\n";
}


cv::Mat YoloPose::letterbox(
    const cv::Mat& frame,
    LetterboxInfo& info
) const {

    const float scaleX =
        static_cast<float>(kInputWidth) /
        static_cast<float>(frame.cols);

    const float scaleY =
        static_cast<float>(kInputHeight) /
        static_cast<float>(frame.rows);

    info.scale =
        std::min(scaleX, scaleY);

    const int resizedWidth =
        static_cast<int>(
            std::round(
                frame.cols *
                info.scale
            )
        );

    const int resizedHeight =
        static_cast<int>(
            std::round(
                frame.rows *
                info.scale
            )
        );

    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(
            resizedWidth,
            resizedHeight
        )
    );

    info.padX =
        (kInputWidth - resizedWidth) / 2;

    info.padY =
        (kInputHeight - resizedHeight) / 2;

    cv::Mat result(
        kInputHeight,
        kInputWidth,
        CV_8UC3,
        cv::Scalar(
            114,
            114,
            114
        )
    );

    resized.copyTo(
        result(
            cv::Rect(
                info.padX,
                info.padY,
                resizedWidth,
                resizedHeight
            )
        )
    );

    return result;
}


std::vector<YoloPoseDetection>
YoloPose::estimate(
    const cv::Mat& bgrFrame
) {

    if (bgrFrame.empty()) {
        throw std::invalid_argument(
            "no se puede estimar pose "
            "en un frame vacio"
        );
    }

    LetterboxInfo letterboxInfo{};

    cv::Mat processed =
        letterbox(
            bgrFrame,
            letterboxInfo
        );

    cv::Mat rgb;

    cv::cvtColor(
        processed,
        rgb,
        cv::COLOR_BGR2RGB
    );

    cv::Mat floatImage;

    rgb.convertTo(
        floatImage,
        CV_32FC3,
        1.0 / 255.0
    );

    const std::size_t imageArea =
        static_cast<std::size_t>(
            kInputWidth *
            kInputHeight
        );

    std::vector<float> inputData(
        3 * imageArea
    );

    // HWC -> CHW
    for (
        int y = 0;
        y < kInputHeight;
        ++y
    ) {
        const cv::Vec3f* row =
            floatImage.ptr<cv::Vec3f>(y);

        for (
            int x = 0;
            x < kInputWidth;
            ++x
        ) {
            const std::size_t pixel =
                static_cast<std::size_t>(
                    y * kInputWidth + x
                );

            inputData[
                0 * imageArea + pixel
            ] = row[x][0];

            inputData[
                1 * imageArea + pixel
            ] = row[x][1];

            inputData[
                2 * imageArea + pixel
            ] = row[x][2];
        }
    }

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputData.data(),
            inputData.size(),
            inputShape_.data(),
            inputShape_.size()
        );

    const char* inputNames[] = {
        inputName_.c_str()
    };

    const char* outputNames[] = {
        outputName_.c_str()
    };

    try {

        auto outputs =
            session_.Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1
            );

        if (outputs.empty()) {
            throw std::runtime_error(
                "YOLO Pose no devolvio salida"
            );
        }

        return decodeOutput(
            outputs.front(),
            letterboxInfo,
            bgrFrame.cols,
            bgrFrame.rows
        );

    } catch (
        const Ort::Exception& error
    ) {

        throw std::runtime_error(
            std::string(
                "fallo durante inferencia YOLO: "
            ) +
            error.what()
        );
    }
}


std::vector<YoloPoseDetection>
YoloPose::decodeOutput(
    const Ort::Value& output,
    const LetterboxInfo& letterboxInfo,
    int originalWidth,
    int originalHeight
) const {

    const float* data =
        output.GetTensorData<float>();

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    std::vector<
        std::vector<Keypoint>
    > candidateKeypoints;

    for (
        int candidate = 0;
        candidate < kCandidateCount;
        ++candidate
    ) {

        const auto getValue =
            [&](int feature) {
                return data[
                    feature *
                    kCandidateCount +
                    candidate
                ];
            };

        const float confidence =
            getValue(4);

        if (
            confidence <
            confThreshold_
        ) {
            continue;
        }

        const float cx =
            getValue(0);

        const float cy =
            getValue(1);

        const float width =
            getValue(2);

        const float height =
            getValue(3);

        float x1 =
            cx - width / 2.0F;

        float y1 =
            cy - height / 2.0F;

        float x2 =
            cx + width / 2.0F;

        float y2 =
            cy + height / 2.0F;

        // Deshacer letterbox
        x1 =
            (
                x1 -
                letterboxInfo.padX
            ) /
            letterboxInfo.scale;

        y1 =
            (
                y1 -
                letterboxInfo.padY
            ) /
            letterboxInfo.scale;

        x2 =
            (
                x2 -
                letterboxInfo.padX
            ) /
            letterboxInfo.scale;

        y2 =
            (
                y2 -
                letterboxInfo.padY
            ) /
            letterboxInfo.scale;

        x1 = std::clamp(
            x1,
            0.0F,
            static_cast<float>(
                originalWidth - 1
            )
        );

        y1 = std::clamp(
            y1,
            0.0F,
            static_cast<float>(
                originalHeight - 1
            )
        );

        x2 = std::clamp(
            x2,
            0.0F,
            static_cast<float>(
                originalWidth - 1
            )
        );

        y2 = std::clamp(
            y2,
            0.0F,
            static_cast<float>(
                originalHeight - 1
            )
        );

        const cv::Rect box(
            cvRound(x1),
            cvRound(y1),
            std::max(
                1,
                cvRound(
                    x2 - x1
                )
            ),
            std::max(
                1,
                cvRound(
                    y2 - y1
                )
            )
        );

        std::vector<Keypoint>
            keypoints;

        keypoints.reserve(
            kKeypointCount
        );

        for (
            int keypointIndex = 0;
            keypointIndex <
            kKeypointCount;
            ++keypointIndex
        ) {

            const int base =
                5 +
                keypointIndex * 3;

            float x =
                getValue(base);

            float y =
                getValue(
                    base + 1
                );

            const float score =
                getValue(
                    base + 2
                );

            // Deshacer letterbox
            x =
                (
                    x -
                    letterboxInfo.padX
                ) /
                letterboxInfo.scale;

            y =
                (
                    y -
                    letterboxInfo.padY
                ) /
                letterboxInfo.scale;

            x = std::clamp(
                x,
                0.0F,
                static_cast<float>(
                    originalWidth - 1
                )
            );

            y = std::clamp(
                y,
                0.0F,
                static_cast<float>(
                    originalHeight - 1
                )
            );

            // Guardamos normalizado 0-1,
            // igual que MoveNet.
            keypoints.push_back({
                x /
                    static_cast<float>(
                        originalWidth
                    ),

                y /
                    static_cast<float>(
                        originalHeight
                    ),

                score
            });
        }

        boxes.push_back(box);

        scores.push_back(
            confidence
        );

        candidateKeypoints.push_back(
            std::move(
                keypoints
            )
        );
    }

    const std::vector<int> keep =
        nonMaximumSuppression(
            boxes,
            scores
        );

    std::vector<
        YoloPoseDetection
    > detections;

    detections.reserve(
        keep.size()
    );

    for (
        const int index :
        keep
    ) {

        detections.push_back({
            boxes[index],
            scores[index],
            candidateKeypoints[index]
        });
    }

    return detections;
}


std::vector<int>
YoloPose::nonMaximumSuppression(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores
) const {

    std::vector<int> indices(
        boxes.size()
    );

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

    std::sort(
        indices.begin(),
        indices.end(),
        [&](int a, int b) {
            return scores[a] >
                   scores[b];
        }
    );

    std::vector<int> keep;

    while (
        !indices.empty()
    ) {

        const int current =
            indices.front();

        keep.push_back(
            current
        );

        std::vector<int>
            remaining;

        for (
            std::size_t i = 1;
            i < indices.size();
            ++i
        ) {

            const int other =
                indices[i];

            const float iou =
                intersectionOverUnion(
                    boxes[current],
                    boxes[other]
                );

            if (
                iou <
                nmsThreshold_
            ) {
                remaining.push_back(
                    other
                );
            }
        }

        indices =
            std::move(
                remaining
            );
    }

    return keep;
}


void YoloPose::drawPoses(
    cv::Mat& frame,
    const std::vector<
        YoloPoseDetection
    >& detections,
    float keypointThreshold
) const {

    static constexpr
    std::array<
        std::array<int, 2>,
        16
    > connections{{
        {{0, 1}},
        {{0, 2}},
        {{1, 3}},
        {{2, 4}},
        {{5, 6}},
        {{5, 7}},
        {{7, 9}},
        {{6, 8}},
        {{8, 10}},
        {{5, 11}},
        {{6, 12}},
        {{11, 12}},
        {{11, 13}},
        {{13, 15}},
        {{12, 14}},
        {{14, 16}}
    }};

    for (
        const auto& detection :
        detections
    ) {

        cv::rectangle(
            frame,
            detection.box,
            cv::Scalar(
                255,
                0,
                0
            ),
            2
        );

        const auto toPoint =
            [&frame](
                const Keypoint& keypoint
            ) {

            return cv::Point(
                cvRound(
                    keypoint.x *
                    static_cast<float>(
                        frame.cols - 1
                    )
                ),

                cvRound(
                    keypoint.y *
                    static_cast<float>(
                        frame.rows - 1
                    )
                )
            );
        };

        for (
            const auto& connection :
            connections
        ) {

            const Keypoint& first =
                detection.keypoints[
                    connection[0]
                ];

            const Keypoint& second =
                detection.keypoints[
                    connection[1]
                ];

            if (
                first.score >
                    keypointThreshold &&
                second.score >
                    keypointThreshold
            ) {

                cv::line(
                    frame,
                    toPoint(first),
                    toPoint(second),
                    cv::Scalar(
                        0,
                        255,
                        255
                    ),
                    2,
                    cv::LINE_AA
                );
            }
        }

        for (
            const Keypoint& keypoint :
            detection.keypoints
        ) {

            if (
                keypoint.score >
                keypointThreshold
            ) {

                cv::circle(
                    frame,
                    toPoint(
                        keypoint
                    ),
                    4,
                    cv::Scalar(
                        0,
                        0,
                        255
                    ),
                    cv::FILLED,
                    cv::LINE_AA
                );
            }
        }
    }
}