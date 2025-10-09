#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{
// Simple logger that keeps TensorRT quiet unless errors occur.
class SampleLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
    {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};

template<typename T>
struct TrtDeleter
{
  void operator()(T* ptr) const noexcept
  {
    delete ptr;
  }
};

template<typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

std::vector<char> readEngineFile(const std::string& path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
  {
    throw std::runtime_error("Failed to open engine file: " + path);
  }

  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  std::vector<char> data(static_cast<std::size_t>(size));
  if (!stream.read(data.data(), size))
  {
    throw std::runtime_error("Failed to read engine bytes from: " + path);
  }

  return data;
}

std::string toString(nvinfer1::DataType type)
{
  switch (type)
  {
    case nvinfer1::DataType::kFLOAT:
      return "FP32";
    case nvinfer1::DataType::kHALF:
      return "FP16";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
    case nvinfer1::DataType::kFP8:
      return "FP8";
    case nvinfer1::DataType::kBF16:
      return "BF16";
    case nvinfer1::DataType::kINT64:
      return "INT64";
    case nvinfer1::DataType::kINT4:
      return "INT4";
    case nvinfer1::DataType::kFP4:
      return "FP4";
    default:
      return "UNKNOWN";
  }
}

std::string dimsToString(const nvinfer1::Dims& dims)
{
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < dims.nbDims; ++i)
  {
    if (i != 0)
    {
      oss << " x ";
    }
    if (dims.d[i] < 0)
    {
      oss << "?";
    }
    else
    {
      oss << dims.d[i];
    }
  }
  oss << "]";
  return oss.str();
}

std::optional<std::size_t> elementCount(const nvinfer1::Dims& dims)
{
  if (dims.nbDims <= 0)
  {
    return std::nullopt;
  }

  std::size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i)
  {
    if (dims.d[i] < 0)
    {
      return std::nullopt;
    }
    count *= static_cast<std::size_t>(dims.d[i]);
  }
  return count;
}

bool hasDynamicDimension(const nvinfer1::Dims& dims)
{
  if (dims.nbDims <= 0)
  {
    return true;
  }

  for (int32_t i = 0; i < dims.nbDims; ++i)
  {
    if (dims.d[i] < 0)
    {
      return true;
    }
  }
  return false;
}

std::size_t elementSize(nvinfer1::DataType type)
{
  switch (type)
  {
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
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT4:
      return 1; // packed representation
    case nvinfer1::DataType::kFP4:
      return 1; // packed representation
    default:
      return 0;
  }
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <path_to_engine_file>" << std::endl;
    return 1;
  }

  const std::string enginePath(argv[1]);

  int deviceCount = 0;
  const cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
  if (cudaStatus != cudaSuccess || deviceCount == 0)
  {
    std::cerr << "[CUDA] No CUDA-capable device found. cudaGetDeviceCount returned "
              << cudaStatus << std::endl;
    return 1;
  }

  try
  {
    SampleLogger logger;
    // Ensure plugins are registered before deserializing the engine.
    if (!initLibNvInferPlugins(&logger, ""))
    {
      std::cerr << "Failed to initialize TensorRT plugins." << std::endl;
      return 1;
    }

    std::vector<char> engineData = readEngineFile(enginePath);

    auto runtime = TrtPtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
      std::cerr << "Failed to create TensorRT runtime." << std::endl;
      return 1;
    }

    auto engine = TrtPtr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine)
    {
      std::cerr << "Failed to deserialize engine from: " << enginePath << std::endl;
      return 1;
    }

    auto context = TrtPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
      std::cerr << "Failed to create execution context." << std::endl;
      return 1;
    }

    std::cout << "Loaded TensorRT engine: " << enginePath << std::endl;
    std::cout << " - Device count available: " << deviceCount << std::endl;
    std::cout << " - Number of tensors (inputs + outputs): " << engine->getNbIOTensors() << std::endl;
    std::cout << " - Number of optimization profiles: " << engine->getNbOptimizationProfiles() << std::endl;

    for (int32_t tensorIndex = 0; tensorIndex < engine->getNbIOTensors(); ++tensorIndex)
    {
      const char* name = engine->getIOTensorName(tensorIndex);
      const bool isInput = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
      nvinfer1::Dims dims = engine->getTensorShape(name);
      if (hasDynamicDimension(dims))
      {
        // Attempt to read dynamic shape from the execution context if available.
        dims = context->getTensorShape(name);
      }

      const nvinfer1::DataType dtype = engine->getTensorDataType(name);
      const auto countOpt = elementCount(dims);
      const std::size_t typeSize = elementSize(dtype);

      std::cout << "   • " << (isInput ? "Input" : "Output") << " tensor '" << name << "'\n";
      std::cout << "     Shape: " << dimsToString(dims) << "\n";
      std::cout << "     Data type: " << toString(dtype) << " (" << typeSize << " bytes)\n";
      if (countOpt.has_value())
      {
        std::cout << "     Elements per tensor: " << countOpt.value() << "\n";
        std::cout << "     Approx. size: " << countOpt.value() * typeSize << " bytes\n";
      }
      else
      {
        std::cout << "     Elements per tensor: dynamic (set at runtime)\n";
      }
    }

    std::cout << "TensorRT engine inspection finished successfully." << std::endl;
  }
  catch (const std::exception& ex)
  {
    std::cerr << "Exception: " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}
