#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtdetrv2_tensorrt_node/label_map.hpp"

namespace rtdetrv2_tensorrt
{

class TrtRtdetrv2Engine;

struct DebugDetection
{
  cv::Rect roi;
  float score{0.0F};
  int label{0};
};

class Rtdetrv2TensorRtNode : public rclcpp::Node
{
public:
  explicit Rtdetrv2TensorRtNode(const rclcpp::NodeOptions & options);
  ~Rtdetrv2TensorRtNode() override;

private:
  void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);
  bool decodeCompressedImage(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg, cv::Mat & out_image) const;
  bool prepareInputTensor(const cv::Mat & image, std::array<int64_t, 2> & orig_size);
  std::vector<DebugDetection> publishDetections(
    const std::vector<float> & boxes, const std::vector<float> & scores,
    const std::vector<int64_t> & labels, const std_msgs::msg::Header & header, int original_width,
    int original_height);
  void annotateDetections(cv::Mat & image, const std::vector<DebugDetection> & detections) const;
  bool encodeAndPublish(const cv::Mat & image, const std_msgs::msg::Header & header);
  void configureEncodingParameters();

  std::unique_ptr<TrtRtdetrv2Engine> engine_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_image_pub_;
  rclcpp::Publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr objects_pub_;

  LabelMapProvider label_map_;
  std::vector<float> input_buffer_;
  cv::Mat blob_buffer_;
  double score_threshold_{0.5};
  int input_width_{640};
  int input_height_{640};
  std::string debug_encoding_;
  bool draw_scores_{true};
  bool draw_class_{true};
  int jpeg_quality_{90};
  bool performance_logging_{false};
  bool debug_visualization_{false};
  std::string encode_extension_;
  std::vector<int> encode_params_;
  std::uint64_t logged_frame_count_{0};
  double preprocess_sum_ms_{0.0};
  double inference_sum_ms_{0.0};
  double postprocess_sum_ms_{0.0};
  double debug_sum_ms_{0.0};
  double total_sum_ms_{0.0};
};

}  // namespace rtdetrv2_tensorrt
