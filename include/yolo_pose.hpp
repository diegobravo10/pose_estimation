#ifndef YOLO_POSE_HPP
#define YOLO_POSE_HPP

#include "pose_estimator.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct YoloPoseDetection {
    cv::Rect box;
    float score;
    std::vector<Keypoint> keypoints;
};

class YoloPose {
public:
    explicit YoloPose(
        const std::string& modelPath =
            "model/yolov8n-pose.onnx"
    );

    std::vector<YoloPoseDetection> estimate(
        const cv::Mat& bgrFrame
    );

    void drawPoses(
        cv::Mat& frame,
        const std::vector<YoloPoseDetection>& detections,
        float keypointThreshold = 0.3F
    ) const;

private:
    static constexpr int kInputWidth = 640;
    static constexpr int kInputHeight = 640;
    static constexpr int kKeypointCount = 17;
    static constexpr int kFeatureCount = 56;
    static constexpr int kCandidateCount = 8400;

    std::string modelPath_;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::Session session_;

    std::string inputName_;
    std::string outputName_;

    std::vector<std::int64_t> inputShape_;

    float confThreshold_ = 0.40F;
    float nmsThreshold_ = 0.45F;

    struct LetterboxInfo {
        float scale;
        int padX;
        int padY;
    };

    cv::Mat letterbox(
        const cv::Mat& frame,
        LetterboxInfo& info
    ) const;

    std::vector<YoloPoseDetection> decodeOutput(
        const Ort::Value& output,
        const LetterboxInfo& letterboxInfo,
        int originalWidth,
        int originalHeight
    ) const;

    std::vector<int> nonMaximumSuppression(
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores
    ) const;
};

#endif