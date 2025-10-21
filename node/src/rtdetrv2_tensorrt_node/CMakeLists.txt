cmake_minimum_required(VERSION 3.16)
project(rtdetrv2_tensorrt_node LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(tier4_perception_msgs REQUIRED)
find_package(autoware_perception_msgs REQUIRED)

add_executable(trt_engine_info src/trt_engine_info.cpp)
target_include_directories(trt_engine_info PRIVATE ${TensorRT_INCLUDE_DIRS})
target_link_libraries(trt_engine_info PRIVATE ${TensorRT_LIBRARIES} CUDA::cudart)

add_library(rtdetrv2_tensorrt_engine SHARED
  src/trt_rtdetrv2_engine.cpp)
target_include_directories(
  rtdetrv2_tensorrt_engine
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_include_directories(
  rtdetrv2_tensorrt_engine
  PRIVATE
  ${TensorRT_INCLUDE_DIRS})

target_link_libraries(
  rtdetrv2_tensorrt_engine
  PUBLIC
  ${TensorRT_LIBRARIES}
  CUDA::cudart)

add_library(rtdetrv2_tensorrt_node SHARED
  src/rtdetrv2_tensorrt_node.cpp
  src/label_map.cpp)
target_include_directories(
  rtdetrv2_tensorrt_node
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_include_directories(
  rtdetrv2_tensorrt_node
  PRIVATE
  ${TensorRT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(
  rtdetrv2_tensorrt_node
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  tier4_perception_msgs
  autoware_perception_msgs)
  
target_link_libraries(
  rtdetrv2_tensorrt_node
  rtdetrv2_tensorrt_engine
  ${OpenCV_LIBRARIES}
  CUDA::cudart)

rclcpp_components_register_node(
  rtdetrv2_tensorrt_node
  PLUGIN "rtdetrv2_tensorrt::Rtdetrv2TensorRtNode"
  EXECUTABLE rtdetrv2_tensorrt_inference)

install(
  TARGETS
  trt_engine_info
  rtdetrv2_tensorrt_engine
  rtdetrv2_tensorrt_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  tier4_perception_msgs
  autoware_perception_msgs
  OpenCV
  TensorRT
  CUDAToolkit)

ament_package()
