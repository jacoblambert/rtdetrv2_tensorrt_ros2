# RTDETRv2 TensorRT Node

An ROS 2 component node that runs NVIDIA TensorRT-accelerated RT-DETRv2 object detection, publishes Autoware-compatible detections, and (optionally) emits annotated debug imagery and performance modules.

## Prerequisites

- ROS 2 Humble
- NVIDIA TensorRT 10+
- CUDA-compatible GPU and drivers

This package vendors the necessary Autoware message definitions (`autoware_perception_msgs`, `tier4_perception_msgs`). No additional message repositories are required.

## Build

```bash
cd /workspace  # your colcon workspace root
source /opt/ros/humble/setup.bash
colcon build --packages-up-to rtdetrv2_tensorrt_node --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"
```

Source the resulting overlay before launching:

```bash
source install/setup.bash
```

## Run

```bash
ros2 launch rtdetrv2_tensorrt_node rtdetrv2_tensorrt.launch.xml \
  engine_path:=/path/to/model.engine \
  label_map_name:=autoware
```

Common label map presets:

- `autoware` (default) – 8 driving-centric classes + traffic light and traffic sign (10 total)
- `coco` – COCO-80 classes (unsupported classes map to `UNKNOWN`)
- Any file path – plain-text label list (one label per line)

## Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `~/in/image/compressed` (configurable via `input_topic`) | `sensor_msgs/msg/CompressedImage` | Subscription | Source images (BGR JPEG/PNG). |
| `~/out/objects` (configurable via `objects_topic`) | `tier4_perception_msgs/msg/DetectedObjectsWithFeature` | Publication | Detected objects with ROI metadata. |
| `~/out/image/compressed` (configurable via `debug_image_topic`) | `sensor_msgs/msg/CompressedImage` | Publication (optional) | Annotated debug image; only published when `debug_visualization` is `true`. |

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `engine_path` | `string` | `""` (required) | Absolute path to the TensorRT engine (`.engine`) file. |
| `input_topic` | `string` | `"~/in/image/compressed"` | Subscribed compressed image topic. |
| `objects_topic` | `string` | `"~/out/objects"` | Published detections topic. |
| `debug_image_topic` | `string` | `"~/out/image/compressed"` | Published annotated image topic (if enabled). |
| `score_threshold` | `double` | `0.5` | Minimum confidence for a detection to be emitted. |
| `debug_encoding` | `string` | `"jpeg"` | Encoding used for debug images (`jpeg` or `png`). |
| `draw_score` | `bool` | `true` | Toggle confidence text overlay on debug images. |
| `draw_class` | `bool` | `true` | Toggle class label text overlay on debug images. |
| `jpeg_quality` | `int` | `90` | JPEG quality when `debug_encoding` is `jpeg` (0–100). |
| `performance_logging` | `bool` | `false` | Enable per-frame and rolling-average timing logs (ms + GPU memory). |
| `debug_visualization` | `bool` | `false` | If `true`, publish annotated debug images. |
| `warmup_iterations` | `int` | `1` | Number of warm-up inferences executed at startup (≥ 0). |
| `label_map_name` | `string` | `"autoware"` | Class mapping preset (`autoware`, `coco`, or file path). |

## Tips
- When `debug_visualization` is enabled, consider lowering `jpeg_quality` or switching to `png` if you need minimal compression artifacts.
- For deterministic latency measurements, enable `performance_logging` and review the per-frame timing output in the node’s log stream.
