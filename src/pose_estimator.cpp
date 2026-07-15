#include "pose_estimator.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

std::string resolveModelPath(const std::string& requestedPath) {
    if (std::filesystem::exists(requestedPath)) {
        return requestedPath;
    }

    if (requestedPath == "models/movenet_singlepose_lightning.onnx" &&
        std::filesystem::exists("model/movenet_singlepose_lightning.onnx")) {
        std::cerr << "Advertencia: se usara la ruta existente: "
                  << "model/movenet_singlepose_lightning.onnx\n";
        return "model/movenet_singlepose_lightning.onnx";
    }

    throw std::runtime_error("modelo no encontrado: " + requestedPath);
}

Ort::SessionOptions makeSessionOptions() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(2);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

Ort::Session makeSession(
    const Ort::Env& env,
    const std::string& modelPath,
    const Ort::SessionOptions& options
) {
    try {
        return Ort::Session(env, modelPath.c_str(), options);
    } catch (const Ort::Exception& error) {
        throw std::runtime_error(
            std::string("error al cargar el modelo con ONNX Runtime: ") +
            error.what()
        );
    }
}

std::string shapeToString(const std::vector<std::int64_t>& shape) {
    std::ostringstream text;
    text << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text << ',';
        }
        text << shape[i];
    }
    text << ']';
    return text.str();
}

}  // namespace

PoseEstimator::PoseEstimator(const std::string& modelPath)
    : modelPath_(resolveModelPath(modelPath)),
      env_(ORT_LOGGING_LEVEL_WARNING, "tesis_pose"),
      sessionOptions_(makeSessionOptions()),
      session_(makeSession(env_, modelPath_, sessionOptions_)),
      inputType_(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED),
      inputIsNhwc_(false),
      inputWidth_(0),
      inputHeight_(0) {
    try {
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() < 1) {
            throw std::runtime_error(
                "se esperaba un modelo con una entrada y al menos una salida"
            );
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const auto inputName = session_.GetInputNameAllocated(0, allocator);
        const auto outputName = session_.GetOutputNameAllocated(0, allocator);
        inputName_ = inputName.get();
        outputName_ = outputName.get();

        const auto inputTypeInfo = session_.GetInputTypeInfo(0);
        const auto outputTypeInfo = session_.GetOutputTypeInfo(0);
        const auto inputInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const auto outputInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape_ = inputInfo.GetShape();
        outputShape_ = outputInfo.GetShape();
        inputType_ = inputInfo.GetElementType();

        if (inputShape_.size() != 4 || inputShape_[0] != 1) {
            throw std::runtime_error(
                "forma de entrada no soportada: " + shapeToString(inputShape_)
            );
        }

        if (inputShape_[3] == 3) {
            inputIsNhwc_ = true;
            inputHeight_ = static_cast<int>(inputShape_[1]);
            inputWidth_ = static_cast<int>(inputShape_[2]);
        } else if (inputShape_[1] == 3) {
            inputIsNhwc_ = false;
            inputHeight_ = static_cast<int>(inputShape_[2]);
            inputWidth_ = static_cast<int>(inputShape_[3]);
        } else {
            throw std::runtime_error(
                "la entrada no es NHWC ni NCHW con 3 canales: " +
                shapeToString(inputShape_)
            );
        }

        if (inputWidth_ <= 0 || inputHeight_ <= 0) {
            throw std::runtime_error("el modelo tiene dimensiones espaciales dinamicas");
        }

        if (inputType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            inputType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            throw std::runtime_error(
                "solo se soportan entradas tensor(float) y tensor(int32)"
            );
        }

        std::cout << "Modelo cargado correctamente.\n"
                  << "Runtime: ONNX Runtime\n"
                  << "Execution Provider: CPU\n"
                  << "Input name: " << inputName_ << '\n'
                  << "Input shape: " << shapeToString(inputShape_) << '\n'
                  << "Output name: " << outputName_ << '\n'
                  << "Output shape: " << shapeToString(outputShape_) << '\n';
    } catch (const Ort::Exception& error) {
        throw std::runtime_error(
            std::string("error al consultar el modelo ONNX: ") + error.what()
        );
    }
}

std::vector<Keypoint> PoseEstimator::estimate(const cv::Mat& bgrFrame) {
    if (bgrFrame.empty()) {
        throw std::invalid_argument("no se puede estimar pose en un frame vacio");
    }

    cv::Mat rgbFrame;
    cv::cvtColor(bgrFrame, rgbFrame, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgbFrame, resized, cv::Size(inputWidth_, inputHeight_));
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }

    const std::size_t pixelCount =
        static_cast<std::size_t>(inputWidth_) * inputHeight_;
    const std::size_t elementCount = pixelCount * 3;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );
    Ort::Value inputTensor{nullptr};

    std::vector<std::int32_t> intInput;
    std::vector<float> floatInput;

    const auto fillInput = [&](auto& destination) {
        using ValueType = typename std::decay_t<decltype(destination)>::value_type;
        destination.resize(elementCount);
        const unsigned char* pixels = resized.ptr<unsigned char>();

        if (inputIsNhwc_) {
            std::transform(
                pixels, pixels + elementCount, destination.begin(),
                [](unsigned char value) { return static_cast<ValueType>(value); }
            );
        } else {
            for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    destination[channel * pixelCount + pixel] =
                        static_cast<ValueType>(pixels[pixel * 3 + channel]);
                }
            }
        }
    };

    if (inputType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        fillInput(intInput);
        inputTensor = Ort::Value::CreateTensor<std::int32_t>(
            memoryInfo, intInput.data(), intInput.size(),
            inputShape_.data(), inputShape_.size()
        );
    } else {
        fillInput(floatInput);
        inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, floatInput.data(), floatInput.size(),
            inputShape_.data(), inputShape_.size()
        );
    }

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};

    try {
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
            outputNames, 1
        );
        if (outputs.empty()) {
            throw std::runtime_error("ONNX Runtime no devolvio ninguna salida");
        }
        return decodeOutput(outputs.front());
    } catch (const Ort::Exception& error) {
        throw std::runtime_error(
            std::string("fallo durante la inferencia de ONNX Runtime: ") +
            error.what()
        );
    }
}

std::vector<Keypoint> PoseEstimator::decodeOutput(const Ort::Value& output) const {
    if (!output.IsTensor()) {
        throw std::runtime_error("salida inesperada: no es un tensor");
    }

    const auto info = output.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != static_cast<std::size_t>(kKeypointCount * 3) ||
        shape.empty() || shape.back() != 3) {
        throw std::runtime_error(
            "salida inesperada para MoveNet: " + shapeToString(shape)
        );
    }

    const float* values = output.GetTensorData<float>();
    std::vector<Keypoint> keypoints;
    keypoints.reserve(kKeypointCount);

    for (int i = 0; i < kKeypointCount; ++i) {
        keypoints.push_back({
            std::clamp(values[i * 3 + 1], 0.0F, 1.0F),
            std::clamp(values[i * 3], 0.0F, 1.0F),
            values[i * 3 + 2]
        });
    }
    return keypoints;
}

void PoseEstimator::drawPose(
    cv::Mat& frame,
    const std::vector<Keypoint>& keypoints,
    float confidenceThreshold
) const {
    if (frame.empty() || keypoints.size() != kKeypointCount) {
        return;
    }

    static constexpr std::array<std::array<int, 2>, 16> connections{{
        {{0, 1}}, {{0, 2}}, {{1, 3}}, {{2, 4}},
        {{5, 6}}, {{5, 7}}, {{7, 9}}, {{6, 8}},
        {{8, 10}}, {{5, 11}}, {{6, 12}}, {{11, 12}},
        {{11, 13}}, {{13, 15}}, {{12, 14}}, {{14, 16}}
    }};

    const auto toPoint = [&frame](const Keypoint& keypoint) {
        return cv::Point(
            cvRound(keypoint.x * static_cast<float>(frame.cols - 1)),
            cvRound(keypoint.y * static_cast<float>(frame.rows - 1))
        );
    };

    for (const auto& connection : connections) {
        const Keypoint& first = keypoints[connection[0]];
        const Keypoint& second = keypoints[connection[1]];
        if (first.score > confidenceThreshold &&
            second.score > confidenceThreshold) {
            cv::line(frame, toPoint(first), toPoint(second),
                     cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
    }

    for (const Keypoint& keypoint : keypoints) {
        if (keypoint.score > confidenceThreshold) {
            cv::circle(frame, toPoint(keypoint), 4, cv::Scalar(0, 0, 255),
                       cv::FILLED, cv::LINE_AA);
        }
    }
}
