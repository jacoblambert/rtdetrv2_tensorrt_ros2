# SPDX-License-Identifier: Apache-2.0
#
# FindTensorRT
# ------------
#
# Locates the TensorRT libraries and headers.
# The following variables will be defined:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#

if(TensorRT_INCLUDE_DIRS AND TensorRT_LIBRARIES)
  set(TensorRT_FOUND TRUE)
  return()
endif()

set(_tensorrt_roots
  $ENV{TENSORRT_ROOT}
  $ENV{TRT_LIB_PATH}
  /usr
  /usr/local
  /opt/tensorrt
  /opt/nvidia/tensorrt
)

find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_tensorrt_roots}
  PATH_SUFFIXES include include/x86_64-linux-gnu include/x86_64-linux-gnu/NvInfer
)

find_library(TensorRT_nvinfer_LIBRARY
  NAMES nvinfer
  HINTS ${_tensorrt_roots}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

find_library(TensorRT_nvinfer_plugin_LIBRARY
  NAMES nvinfer_plugin
  HINTS ${_tensorrt_roots}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

find_library(TensorRT_nvonnxparser_LIBRARY
  NAMES nvonnxparser
  HINTS ${_tensorrt_roots}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

set(TensorRT_LIBRARIES ${TensorRT_nvinfer_LIBRARY})
if(TensorRT_nvinfer_plugin_LIBRARY)
  list(APPEND TensorRT_LIBRARIES ${TensorRT_nvinfer_plugin_LIBRARY})
endif()
if(TensorRT_nvonnxparser_LIBRARY)
  list(APPEND TensorRT_LIBRARIES ${TensorRT_nvonnxparser_LIBRARY})
endif()

set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY
)

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_LIBRARIES
  TensorRT_nvinfer_LIBRARY
  TensorRT_nvinfer_plugin_LIBRARY
  TensorRT_nvonnxparser_LIBRARY
)
