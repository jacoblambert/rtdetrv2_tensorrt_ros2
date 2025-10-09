#include "rtdetrv2_tensorrt_node/rtdetrv2_tensorrt_node.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

class TensorRtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity > severity_threshold_) {
      return;
    }
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        RCLCPP_ERROR(rclcpp::get_logger("TensorRtLogger"), "%s", msg);
        break;
      case Severity::kWARNING:
        RCLCPP_WARN(rclcpp::get_logger("TensorRtLogger"), "%s", msg);
        break;
      default:
        RCLCPP_INFO(rclcpp::get_logger("TensorRtLogger"), "%s", msg);
        break;
    }
  }

  void setSeverityThreshold(Severity severity) { severity_threshold_ = severity; }

private:
  Severity severity_threshold_{Severity::kERROR};
};

template <typename T>
struct TrtDeleter
{
  void operator()(T * ptr) const noexcept
  {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

inline std::size_t elementSize(nvinfer1::DataType type)
{
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
    case nvinfer1::DataType::kUINT8:
      return 1;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT4:
      return 1;
    case nvinfer1::DataType::kFP4:
      return 1;
    default:
      throw std::runtime_error("Unsupported TensorRT data type encountered.");
  }
}

inline std::size_t tensorVolume(const nvinfer1::Dims & dims)
{
  if (dims.nbDims <= 0) {
    return 0;
  }
  std::size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    const int64_t dim = dims.d[i];
    if (dim < 0) {
      return 0;
    }
    volume *= static_cast<std::size_t>(dim);
  }
  return volume;
}

inline bool hasDynamicDimension(const nvinfer1::Dims & dims)
{
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return true;
    }
  }
  return false;
}

inline nvinfer1::Dims resolveInputDims(const std::string & tensor_name, const nvinfer1::Dims & dims)
{
  nvinfer1::Dims resolved = dims;
  for (int i = 0; i < resolved.nbDims; ++i) {
    if (resolved.d[i] < 0) {
      if (i == 0) {
        resolved.d[i] = 1;
      } else {
        throw std::runtime_error(
          "Dynamic dimension detected in tensor '" + tensor_name +
          "'. Only the batch dimension is supported.");
      }
    }
  }
  return resolved;
}

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

class TrtRtdetrv2Engine
{
public:
  struct InferenceResult
  {
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int64_t> labels;
  };

  explicit TrtRtdetrv2Engine(const std::string & engine_path);
  ~TrtRtdetrv2Engine();

  TrtRtdetrv2Engine(const TrtRtdetrv2Engine &) = delete;
  TrtRtdetrv2Engine & operator=(const TrtRtdetrv2Engine &) = delete;

  bool infer(
    const std::vector<float> & image_tensor, const std::array<int64_t, 2> & orig_size,
    InferenceResult & out_result);

  int inputWidth() const noexcept { return input_width_; }
  int inputHeight() const noexcept { return input_height_; }

private:
  struct TensorBinding
  {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    std::size_t bytes{0};
    void * device_ptr{nullptr};
    bool is_input{false};
  };

  void loadEngine(const std::string & engine_path);
  void allocateBindings();
  void releaseBindings();

  TensorRtLogger logger_;
  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IExecutionContext> context_;

  std::vector<TensorBinding> bindings_;
  TensorBinding* images_binding_{nullptr};
  TensorBinding* orig_sizes_binding_{nullptr};
  TensorBinding* boxes_binding_{nullptr};
  TensorBinding* scores_binding_{nullptr};
  TensorBinding* labels_binding_{nullptr};

  cudaStream_t stream_{nullptr};

  int input_width_{640};
  int input_height_{640};
};

TrtRtdetrv2Engine::TrtRtdetrv2Engine(const std::string & engine_path)
{
  if (engine_path.empty()) {
    throw std::invalid_argument("Engine path is empty.");
  }
  loadEngine(engine_path);
  allocateBindings();
}

TrtRtdetrv2Engine::~TrtRtdetrv2Engine()
{
  try {
    releaseBindings();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TrtRtdetrv2Engine"), "Failed to release TensorRT buffers: %s",
      e.what());
  }
}

bool TrtRtdetrv2Engine::infer(
  const std::vector<float> & image_tensor, const std::array<int64_t, 2> & orig_size,
  InferenceResult & out_result)
{
  if (!images_binding_ || !orig_sizes_binding_ || !boxes_binding_ || !scores_binding_ ||
      !labels_binding_) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TrtRtdetrv2Engine"),
      "Engine bindings are not initialized. Did allocation succeed?");
    return false;
  }

  const std::size_t expected_image_bytes = images_binding_->bytes;
  if (image_tensor.size() * sizeof(float) != expected_image_bytes) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TrtRtdetrv2Engine"),
      "Input tensor size mismatch. Expected %zu floats, received %zu floats.",
      expected_image_bytes / sizeof(float), image_tensor.size());
    return false;
  }

  for (auto & binding : bindings_) {
    if (binding.is_input) {
      if (!context_->setInputShape(binding.name.c_str(), binding.dims)) {
        RCLCPP_ERROR(
          rclcpp::get_logger("TrtRtdetrv2Engine"),
          "Failed to set input shape for tensor '%s'.", binding.name.c_str());
        return false;
      }
    }
    if (!context_->setTensorAddress(binding.name.c_str(), binding.device_ptr)) {
      RCLCPP_ERROR(
        rclcpp::get_logger("TrtRtdetrv2Engine"),
        "Failed to set device address for tensor '%s'.", binding.name.c_str());
      return false;
    }
  }

  cudaCheck(
    cudaMemcpyAsync(
      images_binding_->device_ptr, image_tensor.data(), images_binding_->bytes,
      cudaMemcpyHostToDevice, stream_),
    "cudaMemcpyAsync(images)");

  if (orig_sizes_binding_->bytes != sizeof(int64_t) * 2) {
    RCLCPP_WARN_ONCE(
      rclcpp::get_logger("TrtRtdetrv2Engine"),
      "Expected orig_target_sizes tensor size of 2 int64 elements, got %zu bytes.",
      orig_sizes_binding_->bytes);
  }

  cudaCheck(
    cudaMemcpyAsync(
      orig_sizes_binding_->device_ptr, orig_size.data(), orig_sizes_binding_->bytes,
      cudaMemcpyHostToDevice, stream_),
    "cudaMemcpyAsync(orig_target_sizes)");

  if (!context_->enqueueV3(stream_)) {
    RCLCPP_ERROR(rclcpp::get_logger("TrtRtdetrv2Engine"), "TensorRT execution failed.");
    cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize after failure");
    return false;
  }

  out_result.boxes.resize(boxes_binding_->bytes / sizeof(float));
  out_result.scores.resize(scores_binding_->bytes / sizeof(float));
  out_result.labels.resize(labels_binding_->bytes / sizeof(int64_t));

  cudaCheck(
    cudaMemcpyAsync(
      out_result.boxes.data(), boxes_binding_->device_ptr, boxes_binding_->bytes,
      cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync(boxes)");
  cudaCheck(
    cudaMemcpyAsync(
      out_result.scores.data(), scores_binding_->device_ptr, scores_binding_->bytes,
      cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync(scores)");
  cudaCheck(
    cudaMemcpyAsync(
      out_result.labels.data(), labels_binding_->device_ptr, labels_binding_->bytes,
      cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync(labels)");

  cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

void TrtRtdetrv2Engine::loadEngine(const std::string & engine_path)
{
  std::ifstream engine_stream(engine_path, std::ios::binary);
  if (!engine_stream.good()) {
    throw std::runtime_error("Unable to open TensorRT engine file: " + engine_path);
  }
  engine_stream.seekg(0, std::ifstream::end);
  const size_t size = static_cast<size_t>(engine_stream.tellg());
  engine_stream.seekg(0, std::ifstream::beg);
  std::vector<char> engine_data(size);
  engine_stream.read(engine_data.data(), static_cast<std::streamsize>(size));
  engine_stream.close();

  runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("Failed to create TensorRT runtime.");
  }

  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize TensorRT engine.");
  }

  context_ = TrtUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create TensorRT execution context.");
  }
}

void TrtRtdetrv2Engine::allocateBindings()
{
  const int32_t tensor_count = engine_->getNbIOTensors();
  if (tensor_count <= 0) {
    throw std::runtime_error("TensorRT engine reports zero I/O tensors.");
  }

  bindings_.clear();
  bindings_.reserve(static_cast<std::size_t>(tensor_count));

  for (int32_t i = 0; i < tensor_count; ++i) {
    const char * name = engine_->getIOTensorName(i);
    const bool is_input = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    if (!is_input) {
      continue;
    }
    nvinfer1::Dims dims = engine_->getTensorShape(name);
    if (hasDynamicDimension(dims)) {
      dims = resolveInputDims(name, dims);
    }
    if (!context_->setInputShape(name, dims)) {
      throw std::runtime_error("Failed to set input shape for tensor '" + std::string(name) + "'.");
    }
  }

  for (int32_t i = 0; i < tensor_count; ++i) {
    const char * name_cstr = engine_->getIOTensorName(i);
    std::string name(name_cstr);
    const bool is_input = engine_->getTensorIOMode(name_cstr) == nvinfer1::TensorIOMode::kINPUT;

    nvinfer1::Dims dims = context_->getTensorShape(name_cstr);
    if (tensorVolume(dims) == 0) {
      if (is_input) {
        nvinfer1::Dims resolved = resolveInputDims(name, engine_->getTensorShape(name_cstr));
        if (!context_->setInputShape(name_cstr, resolved)) {
          throw std::runtime_error(
            "Failed to resolve final shape for tensor '" + name + "'.");
        }
        dims = context_->getTensorShape(name_cstr);
      }
      if (tensorVolume(dims) == 0) {
        throw std::runtime_error(
          "Tensor '" + name + "' has unresolved dynamic dimensions after setup.");
      }
    }

    const nvinfer1::DataType dtype = engine_->getTensorDataType(name_cstr);
    const std::size_t bytes = tensorVolume(dims) * elementSize(dtype);
    if (bytes == 0) {
      throw std::runtime_error("Computed zero-sized buffer for tensor '" + name + "'.");
    }

    TensorBinding binding;
    binding.name = name;
    binding.dims = dims;
    binding.dtype = dtype;
    binding.bytes = bytes;
    binding.is_input = is_input;

    std::string malloc_context = "cudaMalloc(" + name + ")";
    cudaCheck(cudaMalloc(&binding.device_ptr, binding.bytes), malloc_context.c_str());

    if (!context_->setTensorAddress(name_cstr, binding.device_ptr)) {
      throw std::runtime_error("Failed to set tensor address for '" + name + "'.");
    }

    bindings_.push_back(binding);
    TensorBinding & stored = bindings_.back();

    if (stored.name == "images") {
      if (stored.dims.nbDims < 4) {
        throw std::runtime_error("Unexpected dimension count for 'images' tensor.");
      }
      input_height_ = static_cast<int>(stored.dims.d[2]);
      input_width_ = static_cast<int>(stored.dims.d[3]);
      images_binding_ = &stored;
    } else if (stored.name == "orig_target_sizes") {
      orig_sizes_binding_ = &stored;
    } else if (stored.name == "boxes") {
      boxes_binding_ = &stored;
    } else if (stored.name == "scores") {
      scores_binding_ = &stored;
    } else if (stored.name == "labels") {
      labels_binding_ = &stored;
    }
  }

  if (!images_binding_ || !orig_sizes_binding_ || !boxes_binding_ || !scores_binding_ ||
      !labels_binding_) {
    throw std::runtime_error("Expected TensorRT engine inputs/outputs were not found.");
  }

  cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

void TrtRtdetrv2Engine::releaseBindings()
{
  for (auto & binding : bindings_) {
    if (binding.device_ptr != nullptr) {
      cudaCheck(cudaFree(binding.device_ptr), "cudaFree");
      binding.device_ptr = nullptr;
    }
  }
  bindings_.clear();

  if (stream_ != nullptr) {
    cudaCheck(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    stream_ = nullptr;
  }
}

namespace
{

std::vector<std::string> defaultClasses()
{
  return {
    "car", "truck", "bus", "trailer", "motorcycle", "bicycle", "pedestrian", "animal"};
}

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

  engine_ = std::make_unique<TrtRtdetrv2Engine>(engine_path);
  input_width_ = engine_->inputWidth();
  input_height_ = engine_->inputHeight();
  input_buffer_.resize(static_cast<std::size_t>(3 * input_height_ * input_width_));

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

  const std::size_t expected_tensor_size =
    static_cast<std::size_t>(3 * input_height_) * static_cast<std::size_t>(input_width_);
  if (input_buffer_.size() != expected_tensor_size) {
    input_buffer_.resize(expected_tensor_size);
  }

  cv::dnn::blobFromImage(
    image, blob_buffer_, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(),
    true, false, CV_32F);

  const std::size_t blob_elements = static_cast<std::size_t>(blob_buffer_.total());
  if (blob_elements != expected_tensor_size) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Unexpected blob size during preprocessing. Expected %zu floats, got %zu floats.",
      expected_tensor_size, blob_elements);
    return false;
  }

  std::memcpy(input_buffer_.data(), blob_buffer_.ptr<float>(), expected_tensor_size * sizeof(float));
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
