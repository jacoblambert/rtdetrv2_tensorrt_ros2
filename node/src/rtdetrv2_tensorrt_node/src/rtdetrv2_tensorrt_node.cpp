#include "rtdetrv2_tensorrt_node/rtdetrv2_tensorrt_node.hpp"

#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{


inline void cudaCheck(const cudaError_t error, const char * context)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(
      std::string("CUDA error at ") + context + ": " + cudaGetErrorString(error));
  }
}

inline double durationToMilliseconds(const std::chrono::steady_clock::duration & duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

inline double bytesToMiB(std::size_t bytes)
{
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

namespace rtdetrv2_tensorrt
{


namespace
{

std::string makeLabelText(
  const std::vector<std::string> & classes, int label_idx, double score, bool draw_class,
  bool draw_score)
{
  std::string text;
  if (draw_class) {
    if (label_idx >= 0 && label_idx < static_cast<int>(classes.size())) {
      text = classes[label_idx];
    } else {
      text = "cls_" + std::to_string(label_idx);
    }
  }
  if (draw_score) {
    if (!text.empty()) {
      text += " ";
    }
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%.2f", score);
    text += buffer;
  }
  return text;
}

}  // namespace

Rtdetrv2TensorRtNode::Rtdetrv2TensorRtNode(const rclcpp::NodeOptions & options)
: Node("rtdetrv2_tensorrt", options)
{
  const std::string engine_path = this->declare_parameter<std::string>("engine_path", "");
  const std::string input_topic =
    this->declare_parameter<std::string>("input_topic", "~/in/image/compressed");
  const std::string objects_topic =
    this->declare_parameter<std::string>("objects_topic", "~/out/objects");
  const std::string debug_image_topic =
    this->declare_parameter<std::string>("debug_image_topic", "~/out/image/compressed");
  score_threshold_ = this->declare_parameter<double>("score_threshold", 0.5);
  debug_encoding_ = this->declare_parameter<std::string>("debug_encoding", "jpeg");
  draw_scores_ = this->declare_parameter<bool>("draw_score", true);
  draw_class_ = this->declare_parameter<bool>("draw_class", true);
  jpeg_quality_ = this->declare_parameter<int>("jpeg_quality", 90);
  performance_logging_ = this->declare_parameter<bool>("performance_logging", false);
  debug_visualization_ = this->declare_parameter<bool>("debug_visualization", false);
  const int warmup_iterations = this->declare_parameter<int>("warmup_iterations", 2);
  const std::string label_map_name =
    this->declare_parameter<std::string>("label_map_name", "autoware");

  if (engine_path.empty()) {
    throw std::runtime_error("Parameter 'engine_path' must be provided.");
  }
  if (jpeg_quality_ < 0 || jpeg_quality_ > 100) {
    throw std::runtime_error("Parameter 'jpeg_quality' must be between 0 and 100.");
  }
  if (warmup_iterations < 0) {
    throw std::runtime_error("Parameter 'warmup_iterations' must be non-negative.");
  }

  configureEncodingParameters();

  if (!label_map_.load(label_map_name, this->get_logger())) {
    throw std::runtime_error("Failed to load label map '" + label_map_name + "'.");
  }

  requested_input_size_ = loadRequestedInputSize();
  engine_ = std::make_unique<TrtRtdetrv2Engine>(engine_path, requested_input_size_);
  updateInputGeometryFromEngine();

  if (warmup_iterations > 0) {
    const auto warmup_start = std::chrono::steady_clock::now();
    std::vector<float> warmup_tensor(input_buffer_.size(), 0.0F);
    const std::array<int64_t, 2> warmup_sizes{
      {static_cast<int64_t>(input_width_), static_cast<int64_t>(input_height_)}};
    TrtRtdetrv2Engine::InferenceResult warmup_result;
    int completed = 0;
    for (int i = 0; i < warmup_iterations; ++i) {
      if (!engine_->infer(warmup_tensor, warmup_sizes, warmup_result)) {
        RCLCPP_WARN(
          this->get_logger(), "Warmup inference %d/%d failed.", i + 1, warmup_iterations);
        break;
      }
      ++completed;
    }
    const auto warmup_end = std::chrono::steady_clock::now();
    if (performance_logging_ && completed > 0) {
      const double total_ms = durationToMilliseconds(warmup_end - warmup_start);
      const double average_ms = total_ms / static_cast<double>(completed);
      RCLCPP_INFO(
        this->get_logger(),
        "Completed %d warmup iteration(s) in %.3f ms (avg %.3f ms).",
        completed, total_ms, average_ms);
    }
  }

  auto qos = rclcpp::SensorDataQoS();
  image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
    input_topic, qos,
    std::bind(&Rtdetrv2TensorRtNode::imageCallback, this, std::placeholders::_1));

  objects_pub_ = this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    objects_topic, 10);

  if (debug_visualization_) {
    debug_image_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      debug_image_topic, 10);
  }

  RCLCPP_INFO(this->get_logger(), "Loaded TensorRT engine.");
  RCLCPP_INFO(this->get_logger(), "RTDETRv2 TensorRT node ready.");
  RCLCPP_INFO(this->get_logger(), "  Engine: %s", engine_path.c_str());
  RCLCPP_INFO(this->get_logger(), "  Input image: %s", input_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "  Objects: %s", objects_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "  Label map: %s", label_map_name.c_str());
  RCLCPP_INFO(this->get_logger(), "  Input size: %dx%d", input_width_, input_height_);
  RCLCPP_INFO(
    this->get_logger(), "  Performance logging: %s", performance_logging_ ? "ON" : "OFF");
  RCLCPP_INFO(
    this->get_logger(), "  Debug visualization: %s", debug_visualization_ ? "ON" : "OFF");
  if (debug_visualization_) {
    RCLCPP_INFO(this->get_logger(), "  Debug image: %s", debug_image_topic.c_str());
  }
}

Rtdetrv2TensorRtNode::~Rtdetrv2TensorRtNode() = default;

void Rtdetrv2TensorRtNode::imageCallback(
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  if (!engine_) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "TensorRT engine not initialized. Dropping frame.");
    return;
  }

  if (msg->data.empty()) {
    RCLCPP_WARN(this->get_logger(), "Received an empty compressed image.");
    return;
  }

  auto log_perf = [&](const char * fmt, auto &&... args) {
    if (!performance_logging_) {
      return;
    }
    RCLCPP_INFO(this->get_logger(), fmt, std::forward<decltype(args)>(args)...);
  };

  const auto preprocess_start = std::chrono::steady_clock::now();

  cv::Mat image;
  if (!decodeCompressedImage(msg, image)) {
    return;
  }

  std::array<int64_t, 2> orig_size{};
  if (!prepareInputTensor(image, orig_size)) {
    return;
  }

  const auto preprocess_end = std::chrono::steady_clock::now();

  size_t free_before = 0;
  size_t total_before = 0;
  if (performance_logging_) {
    cudaCheck(cudaMemGetInfo(&free_before, &total_before), "cudaMemGetInfo(before inference)");
  }

  const auto inference_start = preprocess_end;
  TrtRtdetrv2Engine::InferenceResult result;
  const bool inference_ok = engine_->infer(input_buffer_, orig_size, result);
  const auto inference_end = std::chrono::steady_clock::now();

  size_t free_after = 0;
  size_t total_after = 0;
  if (performance_logging_) {
    cudaCheck(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo(after inference)");
  }

  if (!inference_ok) {
    RCLCPP_ERROR(this->get_logger(), "Inference failed. Skipping frame.");
    return;
  }

  const auto postprocess_start = inference_end;

  const auto debug_detections = publishDetections(
    result.boxes, result.scores, result.labels, msg->header, static_cast<int>(orig_size[0]),
    static_cast<int>(orig_size[1]));

  const auto postprocess_end = std::chrono::steady_clock::now();

  const auto debug_start = postprocess_end;
  if (debug_visualization_) {
    cv::Mat annotated = image;
    annotateDetections(annotated, debug_detections);
    if (!encodeAndPublish(annotated, msg->header)) {
      return;
    }
  }
  const auto debug_end = std::chrono::steady_clock::now();

  if (performance_logging_) {
    const double preprocess_ms = durationToMilliseconds(preprocess_end - preprocess_start);
    const double inference_ms = durationToMilliseconds(inference_end - inference_start);
    const double postprocess_ms = durationToMilliseconds(postprocess_end - postprocess_start);
    const double debug_ms = durationToMilliseconds(debug_end - debug_start);
    const double total_ms = durationToMilliseconds(postprocess_end - preprocess_start);

    ++logged_frame_count_;
    preprocess_sum_ms_ += preprocess_ms;
    inference_sum_ms_ += inference_ms;
    postprocess_sum_ms_ += postprocess_ms;
    debug_sum_ms_ += debug_ms;
    total_sum_ms_ += total_ms;

    const double total_mib = bytesToMiB(total_after);
    const double used_before_mib = bytesToMiB(total_before - free_before);
    const double used_after_mib = bytesToMiB(total_after - free_after);
    const double delta_mib =
      (static_cast<double>(free_before) - static_cast<double>(free_after)) / (1024.0 * 1024.0);

    log_perf(
      "Timing (ms): preprocess=%.3f inference=%.3f postprocess=%.3f debug=%.3f total=%.3f",
      preprocess_ms, inference_ms, postprocess_ms, debug_ms, total_ms);
    const double denom = static_cast<double>(logged_frame_count_);
    log_perf(
      "Avg timing (ms over %zu frames): preprocess=%.3f inference=%.3f postprocess=%.3f debug=%.3f total=%.3f",
      static_cast<std::size_t>(logged_frame_count_), preprocess_sum_ms_ / denom,
      inference_sum_ms_ / denom, postprocess_sum_ms_ / denom, debug_sum_ms_ / denom, total_sum_ms_ / denom);
    log_perf(
      "GPU memory (MiB): total=%.2f used_before=%.2f used_after=%.2f delta=%.2f",
      total_mib, used_before_mib, used_after_mib, delta_mib);
  }
}

bool Rtdetrv2TensorRtNode::decodeCompressedImage(
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg, cv::Mat & out_image) const
{
  out_image = cv::imdecode(msg->data, cv::IMREAD_COLOR);
  if (out_image.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to decode compressed image.");
    return false;
  }
  return true;
}

bool Rtdetrv2TensorRtNode::prepareInputTensor(
  const cv::Mat & image, std::array<int64_t, 2> & orig_size)
{
  orig_size = {image.cols, image.rows};

  const int target_width = input_width_;
  const int target_height = input_height_;
  if (target_width <= 0 || target_height <= 0) {
    RCLCPP_ERROR(this->get_logger(), "Invalid target input size %dx%d.", target_width, target_height);
    return false;
  }

  const bool needs_resize = (image.cols != target_width) || (image.rows != target_height);
  const cv::Mat * processed = &image;
  if (needs_resize) {
    cv::resize(image, resize_buffer_, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_LINEAR);
    processed = &resize_buffer_;
  }

  const cv::Mat & bgr = *processed;
  if (bgr.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Resized image is empty.");
    return false;
  }
  if (bgr.type() != CV_8UC3) {
    RCLCPP_WARN_ONCE(this->get_logger(), "Unexpected image type %d. Expected CV_8UC3.", bgr.type());
  }

  const std::size_t channel_size = static_cast<std::size_t>(target_width) * target_height;
  const std::size_t expected_tensor_size = channel_size * 3U;
  if (input_buffer_.size() != expected_tensor_size) {
    input_buffer_.resize(expected_tensor_size);
  }

  float * dst_r = input_buffer_.data();
  float * dst_g = dst_r + channel_size;
  float * dst_b = dst_g + channel_size;
  constexpr float inv_255 = 1.0F / 255.0F;

  for (int y = 0; y < target_height; ++y) {
    const cv::Vec3b * row = bgr.ptr<cv::Vec3b>(y);
    const std::size_t row_offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(target_width);
    for (int x = 0; x < target_width; ++x) {
      const cv::Vec3b & pixel = row[x];
      const std::size_t idx = row_offset + static_cast<std::size_t>(x);
      dst_r[idx] = static_cast<float>(pixel[2]) * inv_255;
      dst_g[idx] = static_cast<float>(pixel[1]) * inv_255;
      dst_b[idx] = static_cast<float>(pixel[0]) * inv_255;
    }
  }
  return true;
}

std::vector<DebugDetection> Rtdetrv2TensorRtNode::publishDetections(
  const std::vector<float> & boxes, const std::vector<float> & scores,
  const std::vector<int64_t> & labels, const std_msgs::msg::Header & header, int original_width,
  int original_height)
{
  tier4_perception_msgs::msg::DetectedObjectsWithFeature out_msg;
  out_msg.header = header;

  const std::size_t num_scores = scores.size();
  const std::size_t boxes_stride = 4;

  std::vector<DebugDetection> debug_detections;
  if (debug_visualization_) {
    debug_detections.reserve(num_scores);
  }

  for (std::size_t i = 0; i < num_scores; ++i) {
    const float score = scores.at(i);
    if (score < static_cast<float>(score_threshold_)) {
      continue;
    }

    const auto label_value = (i < labels.size()) ? labels.at(i) : -1;
    const int label_idx = static_cast<int>(label_value);

    const std::size_t base_idx = i * boxes_stride;
    if (base_idx + 3 >= boxes.size()) {
      continue;
    }

    float xmin = boxes.at(base_idx + 0);
    float ymin = boxes.at(base_idx + 1);
    float xmax = boxes.at(base_idx + 2);
    float ymax = boxes.at(base_idx + 3);

    xmin = std::clamp(xmin, 0.0f, static_cast<float>(original_width));
    ymin = std::clamp(ymin, 0.0f, static_cast<float>(original_height));
    xmax = std::clamp(xmax, 0.0f, static_cast<float>(original_width));
    ymax = std::clamp(ymax, 0.0f, static_cast<float>(original_height));

    const int left = std::max(0, static_cast<int>(std::floor(xmin)));
    const int top = std::max(0, static_cast<int>(std::floor(ymin)));
    const int right = std::min(original_width, static_cast<int>(std::ceil(xmax)));
    const int bottom = std::min(original_height, static_cast<int>(std::ceil(ymax)));
    const int roi_width = std::max(0, right - left);
    const int roi_height = std::max(0, bottom - top);
    if (roi_width <= 0 || roi_height <= 0) {
      continue;
    }

    tier4_perception_msgs::msg::DetectedObjectWithFeature object;
    object.feature.roi.x_offset = static_cast<uint32_t>(left);
    object.feature.roi.y_offset = static_cast<uint32_t>(top);
    object.feature.roi.width = static_cast<uint32_t>(roi_width);
    object.feature.roi.height = static_cast<uint32_t>(roi_height);
    object.object.existence_probability = score;
    object.object.classification = label_map_.classificationForLabel(label_idx, score);

    out_msg.feature_objects.push_back(object);

    if (debug_visualization_) {
      DebugDetection debug_detection;
      debug_detection.roi = cv::Rect(left, top, roi_width, roi_height);
      debug_detection.score = score;
      debug_detection.label = label_idx;
      debug_detections.push_back(debug_detection);
    }
  }

  if (objects_pub_) {
    objects_pub_->publish(out_msg);
  }

  return debug_detections;
}

void Rtdetrv2TensorRtNode::annotateDetections(
  cv::Mat & image, const std::vector<DebugDetection> & detections) const
{
  const auto & class_names = label_map_.classNames();
  for (const auto & detection : detections) {
    cv::rectangle(image, detection.roi, cv::Scalar(0, 0, 255), 2);

    const std::string label_text = makeLabelText(
      class_names, detection.label, static_cast<double>(detection.score), draw_class_, draw_scores_);
    if (label_text.empty()) {
      continue;
    }

    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.6;
    const int thickness = 1;
    int baseline = 0;
    const auto text_size = cv::getTextSize(label_text, font_face, font_scale, thickness, &baseline);
    const int text_x = std::max(detection.roi.x, 0);
    const int text_y = std::max(detection.roi.y - 5, text_size.height);
    const cv::Point text_origin(text_x, text_y);

    cv::rectangle(
      image,
      cv::Rect(text_origin.x, text_origin.y - text_size.height, text_size.width, text_size.height + 4),
      cv::Scalar(0, 0, 255), cv::FILLED);
    cv::putText(
      image, label_text, cv::Point(text_origin.x, text_origin.y + 2), font_face, font_scale,
      cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
  }
}

std::optional<TrtRtdetrv2Engine::InputSize> Rtdetrv2TensorRtNode::loadRequestedInputSize()
{
  const int param_width = this->declare_parameter<int>("input_width", 0);
  const int param_height = this->declare_parameter<int>("input_height", 0);

  if (param_width <= 0 && param_height <= 0) {
    return std::nullopt;
  }
  if (param_width <= 0 || param_height <= 0) {
    throw std::runtime_error(
      "Parameters 'input_width' and 'input_height' must both be positive when provided.");
  }

  return TrtRtdetrv2Engine::InputSize{param_width, param_height};
}

void Rtdetrv2TensorRtNode::updateInputGeometryFromEngine()
{
  if (!engine_) {
    throw std::runtime_error("TensorRT engine not initialized when updating input geometry.");
  }

  const int width = engine_->inputWidth();
  const int height = engine_->inputHeight();
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("TensorRT engine reported invalid input dimensions.");
  }

  if (requested_input_size_) {
    if (requested_input_size_->width != width || requested_input_size_->height != height) {
      RCLCPP_WARN(
        this->get_logger(),
        "Requested input size %dx%d differs from engine dimensions %dx%d. Using engine output.",
        requested_input_size_->width, requested_input_size_->height, width, height);
    }
  } else if (!engine_->isInputShapeStatic()) {
    RCLCPP_INFO(
      this->get_logger(),
      "TensorRT engine uses dynamic shapes; resolved to %dx%d for the current profile.",
      width, height);
  }

  input_width_ = width;
  input_height_ = height;

  const std::size_t tensor_size = static_cast<std::size_t>(3 * input_height_) * input_width_;
  if (input_buffer_.size() != tensor_size) {
    input_buffer_.resize(tensor_size);
  }
  resize_buffer_.create(input_height_, input_width_, CV_8UC3);
}

bool Rtdetrv2TensorRtNode::encodeAndPublish(const cv::Mat & image, const std_msgs::msg::Header & header)
{
  if (!debug_image_pub_) {
    return false;
  }
  sensor_msgs::msg::CompressedImage out_msg;
  out_msg.header = header;
  out_msg.format = debug_encoding_;

  if (!cv::imencode(encode_extension_, image, out_msg.data, encode_params_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to encode annotated image.");
    return false;
  }

  debug_image_pub_->publish(std::move(out_msg));
  return true;
}

void Rtdetrv2TensorRtNode::configureEncodingParameters()
{
  encode_params_.clear();
  encode_extension_ = "." + debug_encoding_;

  std::string encoding_lower = debug_encoding_;
  std::transform(encoding_lower.begin(), encoding_lower.end(), encoding_lower.begin(), ::tolower);
  if (encoding_lower == "jpeg" || encoding_lower == "jpg") {
    encode_extension_ = ".jpg";
    encode_params_ = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
  } else if (encoding_lower == "png") {
    encode_extension_ = ".png";
    encode_params_ = {cv::IMWRITE_PNG_COMPRESSION, 1};
  }
}

}  // namespace rtdetrv2_tensorrt

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rtdetrv2_tensorrt::Rtdetrv2TensorRtNode)
