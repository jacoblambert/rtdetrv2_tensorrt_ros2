#pragma once

#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <rclcpp/logger.hpp>

#include <string>
#include <vector>

namespace rtdetrv2_tensorrt
{

class LabelMapProvider
{
public:
  LabelMapProvider() = default;

  bool load(const std::string & map_name, const rclcpp::Logger & logger);

  const std::vector<std::string> & classNames() const noexcept { return class_names_; }

  std::vector<autoware_perception_msgs::msg::ObjectClassification> classificationForLabel(
    int label_index, float score) const;

private:
  bool loadPreset(const std::string & preset, const rclcpp::Logger & logger);
  bool loadAutoware();
  bool loadCoco();
  bool loadFromFile(const std::string & path, const rclcpp::Logger & logger);
  void computeClassificationLookup();

  std::vector<std::string> class_names_;
  std::vector<uint8_t> classification_lookup_;
};

}  // namespace rtdetrv2_tensorrt
