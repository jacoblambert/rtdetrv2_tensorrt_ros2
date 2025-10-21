#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rtdetrv2_tensorrt
{

class TrtRtdetrv2Engine
{
public:
  struct InputSize
  {
    int width{0};
    int height{0};
  };

  struct InferenceResult
  {
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int64_t> labels;
  };

  explicit TrtRtdetrv2Engine(
    const std::string & engine_path,
    std::optional<InputSize> requested_size = std::nullopt);
  ~TrtRtdetrv2Engine();

  TrtRtdetrv2Engine(const TrtRtdetrv2Engine &) = delete;
  TrtRtdetrv2Engine & operator=(const TrtRtdetrv2Engine &) = delete;

  bool infer(
    const std::vector<float> & image_tensor, const std::array<int64_t, 2> & orig_size,
    InferenceResult & out_result);

  int inputWidth() const noexcept;
  int inputHeight() const noexcept;
  bool isInputShapeStatic() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rtdetrv2_tensorrt

