#include "rtdetrv2_tensorrt_node/trt_rtdetrv2_engine.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
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
        std::cerr << "[TensorRT][ERROR] " << msg << std::endl;
        break;
      case Severity::kWARNING:
        std::cerr << "[TensorRT][WARN] " << msg << std::endl;
        break;
      default:
        std::clog << "[TensorRT][INFO] " << msg << std::endl;
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

std::size_t elementSize(nvinfer1::DataType type)
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

std::size_t tensorVolume(const nvinfer1::Dims & dims)
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

bool hasDynamicDimension(const nvinfer1::Dims & dims)
{
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return true;
    }
  }
  return false;
}

nvinfer1::Dims resolveInputDims(
  const std::string & tensor_name, const nvinfer1::Dims & dims,
  const std::optional<rtdetrv2_tensorrt::TrtRtdetrv2Engine::InputSize> & requested_size,
  bool & is_static)
{
  nvinfer1::Dims resolved = dims;
  is_static = true;

  for (int i = 0; i < resolved.nbDims; ++i) {
    if (resolved.d[i] >= 0) {
      continue;
    }

    is_static = false;
    if (i == 0) {
      resolved.d[i] = 1;
      continue;
    }

    if (tensor_name == "images") {
      if (resolved.nbDims < 4) {
        throw std::runtime_error(
          "Tensor '" + tensor_name + "' expected to have at least 4 dimensions.");
      }

      if (!requested_size.has_value()) {
        throw std::runtime_error(
          "Tensor '" + tensor_name + "' has dynamic spatial dimensions. Please set the "
          "input_width and input_height parameters.");
      }

      // Assume NCHW layout for the engine bindings.
      if (i == resolved.nbDims - 2) {
        resolved.d[i] = requested_size->height;
        continue;
      }
      if (i == resolved.nbDims - 1) {
        resolved.d[i] = requested_size->width;
        continue;
      }
    }

    throw std::runtime_error(
      "Dynamic dimension detected in tensor '" + tensor_name +
      "'. Only batch and spatial dimensions of 'images' are supported.");
  }

  return resolved;
}

void cudaCheck(const cudaError_t error, const char * context)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + context + ": " + cudaGetErrorString(error));
  }
}

}  // namespace

namespace rtdetrv2_tensorrt
{

class TrtRtdetrv2Engine::Impl
{
public:
  Impl(const std::string & engine_path, std::optional<InputSize> requested_size)
  : requested_size_(std::move(requested_size))
  {
    if (engine_path.empty()) {
      throw std::invalid_argument("Engine path is empty.");
    }
    loadEngine(engine_path);
    allocateBindings();
  }

  ~Impl()
  {
    try {
      releaseBindings();
    } catch (const std::exception & e) {
      std::cerr << "[TensorRT] Failed to release resources: " << e.what() << std::endl;
    }
  }

  bool infer(
    const std::vector<float> & image_tensor, const std::array<int64_t, 2> & orig_size,
    InferenceResult & out_result)
  {
    if (!images_binding_ || !orig_sizes_binding_ || !boxes_binding_ || !scores_binding_ ||
        !labels_binding_) {
      std::cerr << "[TensorRT] Engine bindings are not initialized." << std::endl;
      return false;
    }

    const std::size_t expected_image_bytes = images_binding_->bytes;
    if (image_tensor.size() * sizeof(float) != expected_image_bytes) {
      std::cerr << "[TensorRT] Input tensor size mismatch. Expected "
                << (expected_image_bytes / sizeof(float)) << " floats, got "
                << image_tensor.size() << " floats." << std::endl;
      return false;
    }

    for (auto & binding : bindings_) {
      if (binding.is_input) {
        if (!context_->setInputShape(binding.name.c_str(), binding.dims)) {
          std::cerr << "[TensorRT] Failed to set input shape for tensor '" << binding.name
                    << "'." << std::endl;
          return false;
        }
      }
      if (!context_->setTensorAddress(binding.name.c_str(), binding.device_ptr)) {
        std::cerr << "[TensorRT] Failed to set device address for tensor '" << binding.name
                  << "'." << std::endl;
        return false;
      }
    }

    cudaCheck(
      cudaMemcpyAsync(
        images_binding_->device_ptr, image_tensor.data(), images_binding_->bytes,
        cudaMemcpyHostToDevice, stream_),
      "cudaMemcpyAsync(images)");

    const auto orig_dtype = orig_sizes_binding_->dtype;
    const std::size_t orig_elem_size = elementSize(orig_dtype);
    const std::size_t orig_element_count =
      orig_elem_size == 0 ? 0 : orig_sizes_binding_->bytes / orig_elem_size;
    if (orig_element_count < 2) {
      std::cerr << "[TensorRT] Unexpected orig_target_sizes tensor element count: "
                << orig_element_count << std::endl;
      return false;
    }

    const void * orig_data_ptr = nullptr;
    std::array<int32_t, 2> orig_size_i32{};
    std::array<float, 2> orig_size_f32{};
    switch (orig_dtype) {
      case nvinfer1::DataType::kINT32:
        orig_size_i32[0] = static_cast<int32_t>(orig_size[0]);
        orig_size_i32[1] = static_cast<int32_t>(orig_size[1]);
        orig_data_ptr = orig_size_i32.data();
        break;
      case nvinfer1::DataType::kFLOAT:
        orig_size_f32[0] = static_cast<float>(orig_size[0]);
        orig_size_f32[1] = static_cast<float>(orig_size[1]);
        orig_data_ptr = orig_size_f32.data();
        break;
      case nvinfer1::DataType::kINT64:
        orig_data_ptr = orig_size.data();
        break;
      default:
        std::cerr << "[TensorRT] Unsupported data type for orig_target_sizes binding: "
                  << static_cast<int>(orig_dtype) << std::endl;
        return false;
    }

    cudaCheck(
      cudaMemcpyAsync(
        orig_sizes_binding_->device_ptr, orig_data_ptr, orig_sizes_binding_->bytes,
        cudaMemcpyHostToDevice, stream_),
      "cudaMemcpyAsync(orig_target_sizes)");

    if (!context_->enqueueV3(stream_)) {
      std::cerr << "[TensorRT] TensorRT execution failed." << std::endl;
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

  int inputWidth() const noexcept { return input_width_; }
  int inputHeight() const noexcept { return input_height_; }
  bool isInputShapeStatic() const noexcept { return input_shape_static_; }

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

  void loadEngine(const std::string & engine_path)
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

  void allocateBindings()
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
        bool dims_static = true;
        dims = resolveInputDims(name, dims, requested_size_, dims_static);
        input_shape_static_ = input_shape_static_ && dims_static;
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
          bool dims_static = true;
          nvinfer1::Dims resolved = resolveInputDims(
            name, engine_->getTensorShape(name_cstr), requested_size_, dims_static);
          input_shape_static_ = input_shape_static_ && dims_static;
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

      cudaCheck(cudaMalloc(&binding.device_ptr, binding.bytes), ("cudaMalloc(" + name + ")").c_str());

      if (!context_->setTensorAddress(name_cstr, binding.device_ptr)) {
        throw std::runtime_error("Failed to set tensor address for '" + name + "'.");
      }

      bindings_.push_back(binding);
      TensorBinding & stored = bindings_.back();

      if (stored.name == "images") {
        if (stored.dims.nbDims < 4) {
          throw std::runtime_error("Unexpected dimension count for 'images' tensor.");
        }
        input_height_ = static_cast<int>(stored.dims.d[stored.dims.nbDims - 2]);
        input_width_ = static_cast<int>(stored.dims.d[stored.dims.nbDims - 1]);
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

  void releaseBindings()
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

  std::optional<InputSize> requested_size_;
  TensorRtLogger logger_;
  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IExecutionContext> context_;

  std::vector<TensorBinding> bindings_;
  TensorBinding * images_binding_{nullptr};
  TensorBinding * orig_sizes_binding_{nullptr};
  TensorBinding * boxes_binding_{nullptr};
  TensorBinding * scores_binding_{nullptr};
  TensorBinding * labels_binding_{nullptr};

  cudaStream_t stream_{nullptr};

  int input_width_{640};
  int input_height_{640};
  bool input_shape_static_{true};
};

TrtRtdetrv2Engine::TrtRtdetrv2Engine(
  const std::string & engine_path, std::optional<InputSize> requested_size)
: impl_(std::make_unique<Impl>(engine_path, std::move(requested_size)))
{
}

TrtRtdetrv2Engine::~TrtRtdetrv2Engine() = default;

bool TrtRtdetrv2Engine::infer(
  const std::vector<float> & image_tensor, const std::array<int64_t, 2> & orig_size,
  InferenceResult & out_result)
{
  return impl_->infer(image_tensor, orig_size, out_result);
}

int TrtRtdetrv2Engine::inputWidth() const noexcept
{
  return impl_->inputWidth();
}

int TrtRtdetrv2Engine::inputHeight() const noexcept
{
  return impl_->inputHeight();
}

bool TrtRtdetrv2Engine::isInputShapeStatic() const noexcept
{
  return impl_->isInputShapeStatic();
}

}  // namespace rtdetrv2_tensorrt
