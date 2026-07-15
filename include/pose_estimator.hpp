#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct Keypoint {
    float x;
    float y;
    float score;
};

class PoseEstimator {
public:
    explicit PoseEstimator(
        const std::string& modelPath =
            "models/movenet_singlepose_lightning.onnx"
    );

    std::vector<Keypoint> estimate(const cv::Mat& bgrFrame);

    void drawPose(
        cv::Mat& frame,
        const std::vector<Keypoint>& keypoints,
        float confidenceThreshold = 0.3F
    ) const;

private:
    static constexpr int kKeypointCount = 17;

    std::string modelPath_;
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::Session session_;
    std::string inputName_;
    std::string outputName_;
    std::vector<std::int64_t> inputShape_;
    std::vector<std::int64_t> outputShape_;
    ONNXTensorElementDataType inputType_;
    bool inputIsNhwc_;
    int inputWidth_;
    int inputHeight_;

    std::vector<Keypoint> decodeOutput(const Ort::Value& output) const;
};

#endif
