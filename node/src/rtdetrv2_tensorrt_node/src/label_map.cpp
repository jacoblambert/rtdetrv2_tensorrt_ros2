#include "rtdetrv2_tensorrt_node/label_map.hpp"

#include <rclcpp/logging.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace rtdetrv2_tensorrt
{
namespace
{

std::string trim(const std::string & input)
{
  const auto begin = std::find_if_not(input.begin(), input.end(), [](unsigned char c) {
    return std::isspace(c) != 0;
  });
  const auto end = std::find_if_not(
    input.rbegin(), input.rend(), [](unsigned char c) { return std::isspace(c) != 0; });
  if (begin >= end.base()) {
    return "";
  }
  return std::string(begin, end.base());
}

std::string toLower(const std::string & input)
{
  std::string lower = input;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lower;
}

std::string normalizeLabel(const std::string & label)
{
  std::string normalized;
  normalized.reserve(label.size());
  for (char ch : label) {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0) {
      normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

uint8_t classificationFromLabel(const std::string & label)
{
  using autoware_perception_msgs::msg::ObjectClassification;
  const std::string normalized = normalizeLabel(label);
  if (normalized.empty()) {
    return ObjectClassification::UNKNOWN;
  }

  static const std::unordered_set<std::string> car_like = {
    "car", "cars", "sedan", "automobile", "vehicle", "van", "minivan"};
  if (car_like.find(normalized) != car_like.end()) {
    return ObjectClassification::CAR;
  }

  static const std::unordered_set<std::string> truck_like = {
    "truck", "pickup", "pickuptruck", "lorry", "cementtruck", "firetruck"};
  if (truck_like.find(normalized) != truck_like.end()) {
    return ObjectClassification::TRUCK;
  }

  static const std::unordered_set<std::string> bus_like = {
    "bus", "coach", "trolleybus", "minibus"};
  if (bus_like.find(normalized) != bus_like.end()) {
    return ObjectClassification::BUS;
  }

  static const std::unordered_set<std::string> trailer_like = {"trailer", "caravan"};
  if (trailer_like.find(normalized) != trailer_like.end()) {
    return ObjectClassification::TRAILER;
  }

  static const std::unordered_set<std::string> motorcycle_like = {
    "motorcycle", "motorbike", "moped", "scooter"};
  if (motorcycle_like.find(normalized) != motorcycle_like.end()) {
    return ObjectClassification::MOTORCYCLE;
  }

  static const std::unordered_set<std::string> bicycle_like = {
    "bicycle", "bike", "cyclist", "cycle"};
  if (bicycle_like.find(normalized) != bicycle_like.end()) {
    return ObjectClassification::BICYCLE;
  }

  static const std::unordered_set<std::string> pedestrian_like = {
    "person", "people", "pedestrian", "walker", "human", "man", "woman"};
  if (pedestrian_like.find(normalized) != pedestrian_like.end()) {
    return ObjectClassification::PEDESTRIAN;
  }

  static const std::unordered_set<std::string> animal_like = {
    "animal", "dog", "cat", "horse", "sheep", "cow", "bear", "elephant", "zebra", "giraffe",
    "bird", "deer", "fox", "monkey", "kangaroo"};
  if (animal_like.find(normalized) != animal_like.end()) {
    return ObjectClassification::ANIMAL;
  }

  static const std::unordered_set<std::string> hazard_like = {
    "trafficcone", "cone", "bollard", "barrier"};
  if (hazard_like.find(normalized) != hazard_like.end()) {
    return ObjectClassification::HAZARD;
  }

  return ObjectClassification::UNKNOWN;
}

const std::vector<std::string> & autowarePreset()
{
  static const std::vector<std::string> names = {
    "car", "truck", "bus", "trailer", "motorcycle", "bicycle", "pedestrian", "animal", "traffic light", "traffic sign"};
  return names;
}

const std::vector<std::string> & cocoPreset()
{
  static const std::vector<std::string> names = {
    "person",      "bicycle",    "car",        "motorcycle", "airplane",   "bus",
    "train",       "truck",      "boat",       "traffic light", "fire hydrant",
    "stop sign",   "parking meter", "bench",  "bird",        "cat",        "dog",
    "horse",       "sheep",      "cow",       "elephant",   "bear",       "zebra",
    "giraffe",     "backpack",   "umbrella",  "handbag",    "tie",        "suitcase",
    "frisbee",     "skis",       "snowboard", "sports ball", "kite",      "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass",  "cup",        "fork",      "knife",      "spoon",      "bowl",
    "banana",      "apple",      "sandwich",  "orange",     "broccoli",   "carrot",
    "hot dog",     "pizza",      "donut",     "cake",       "chair",      "couch",
    "potted plant", "bed",       "dining table", "toilet", "tv",         "laptop",
    "mouse",       "remote",     "keyboard",  "cell phone", "microwave",  "oven",
    "toaster",     "sink",       "refrigerator", "book",   "clock",      "vase",
    "scissors",    "teddy bear", "hair drier", "toothbrush"};
  return names;
}

}  // namespace

bool LabelMapProvider::load(const std::string & map_name, const rclcpp::Logger & logger)
{
  const std::string trimmed = trim(map_name);
  const std::string lower = toLower(trimmed);

  if (lower.empty() || lower == "autoware" || lower == "driving") {
    return loadAutoware();
  }
  if (lower == "coco") {
    return loadCoco();
  }
  if (!loadFromFile(map_name, logger)) {
    return false;
  }
  computeClassificationLookup();
  return true;
}

bool LabelMapProvider::loadPreset(const std::string & preset, const rclcpp::Logger & logger)
{
  const std::string lower = toLower(preset);
  if (lower == "autoware" || lower == "driving") {
    return loadAutoware();
  }
  if (lower == "coco") {
    return loadCoco();
  }
  RCLCPP_WARN(logger, "Unknown label map preset '%s'.", preset.c_str());
  return false;
}

bool LabelMapProvider::loadAutoware()
{
  class_names_ = autowarePreset();
  computeClassificationLookup();
  return true;
}

bool LabelMapProvider::loadCoco()
{
  class_names_ = cocoPreset();
  computeClassificationLookup();
  return true;
}

bool LabelMapProvider::loadFromFile(const std::string & path, const rclcpp::Logger & logger)
{
  const std::filesystem::path file_path(path);
  if (!std::filesystem::exists(file_path)) {
    RCLCPP_ERROR(logger, "Label map file '%s' does not exist.", path.c_str());
    return false;
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    RCLCPP_ERROR(logger, "Unable to open label map file '%s'.", path.c_str());
    return false;
  }

  class_names_.clear();
  std::string line;
  while (std::getline(file, line)) {
    const std::string trimmed_line = trim(line);
    if (!trimmed_line.empty()) {
      class_names_.push_back(trimmed_line);
    }
  }
  if (class_names_.empty()) {
    RCLCPP_ERROR(logger, "Label map file '%s' did not contain any labels.", path.c_str());
    return false;
  }
  return true;
}

void LabelMapProvider::computeClassificationLookup()
{
  classification_lookup_.clear();
  classification_lookup_.reserve(class_names_.size());
  for (const auto & name : class_names_) {
    classification_lookup_.push_back(classificationFromLabel(name));
  }
}

std::vector<autoware_perception_msgs::msg::ObjectClassification>
LabelMapProvider::classificationForLabel(int label_index, float score) const
{
  using autoware_perception_msgs::msg::ObjectClassification;
  ObjectClassification classification;
  classification.label = ObjectClassification::UNKNOWN;
  if (label_index >= 0 && static_cast<std::size_t>(label_index) < classification_lookup_.size()) {
    classification.label = classification_lookup_[static_cast<std::size_t>(label_index)];
  }
  classification.probability = std::clamp(score, 0.0F, 1.0F);
  return {classification};
}

}  // namespace rtdetrv2_tensorrt
