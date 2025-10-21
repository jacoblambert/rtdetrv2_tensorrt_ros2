import os
import argparse
from typing import Optional, Tuple

import onnx
import tensorrt as trt


def _extract_input_hw_from_onnx(onnx_path: str,
                                input_name: str = "images") -> Optional[Tuple[int, int]]:
    """Read the ONNX graph to recover the static spatial size of the main input."""
    model = onnx.load(onnx_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        # Shape inference can fail when opset support is partial; fall back to raw graph.
        pass

    def _dim_value(dim):
        return dim.dim_value if dim.HasField("dim_value") else None

    for value_info in model.graph.input:
        if value_info.name != input_name:
            continue
        tensor_type = value_info.type.tensor_type
        shape = tensor_type.shape.dim
        if len(shape) != 4:
            break
        h_val = _dim_value(shape[2])
        w_val = _dim_value(shape[3])
        if h_val is not None and w_val is not None:
            return int(h_val), int(w_val)
    return None


def main(onnx_path,
         engine_path,
         max_batchsize,
         opt_batchsize,
         min_batchsize,
         input_size,
         use_fp16=True,
         verbose=False) -> None:
    """ Convert ONNX model to TensorRT engine.
    Args:
        onnx_path (str): Path to the input ONNX model.
        engine_path (str): Path to save the output TensorRT engine.
        use_fp16 (bool): Whether to use FP16 precision.
        verbose (bool): Whether to enable verbose logging.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True) # deprecated in trt 10.0

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"[INFO] Loading ONNX file from {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    config = builder.create_builder_config()
    # config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True) # deprecated in trt 10.0
    # config.max_workspace_size = 1 << 30  # 1GB # deprecated in trt 10.0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 optimization enabled.")
        else:
            print(
                "[WARNING] FP16 not supported on this platform. Proceeding with FP32."
            )

    if input_size is None:
        inferred_hw = _extract_input_hw_from_onnx(onnx_path)
        if inferred_hw is None:
            raise ValueError(
                "Unable to infer input spatial size from ONNX. "
                "Specify it explicitly via --inputSize."
            )
        input_hw = inferred_hw
        print(f"[INFO] Detected input size from ONNX: {input_hw[0]}x{input_hw[1]}")
    else:
        input_hw = (input_size, input_size)

    profile = builder.create_optimization_profile()
    profile.set_shape("images",
                      min=(min_batchsize, 3, input_hw[0], input_hw[1]),
                      opt=(opt_batchsize, 3, input_hw[0], input_hw[1]),
                      max=(max_batchsize, 3, input_hw[0], input_hw[1]))
    profile.set_shape("orig_target_sizes",
                      min=(min_batchsize, 2),
                      opt=(opt_batchsize, 2),
                      max=(max_batchsize, 2))
    config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        raise RuntimeError("Failed to build the engine.")

    print(f"[INFO] Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
        # f.write(engine.serialize()) # deprecated in trt 10.0, engine is already serialized
        f.write(engine)
    print("[INFO] Engine export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", "-i", type=str, required=True, help="Path to input ONNX model file")
    parser.add_argument("--inputSize", "-s", type=int, default=None, help="Override network input size if ONNX does not store it")
    parser.add_argument("--saveEngine", "-o", type=str, default="model.engine", help="Path to output TensorRT engine file")
    parser.add_argument("--maxBatchSize", "-Mb", type=int, default=32, help="Maximum batch size for inference")
    parser.add_argument("--optBatchSize", "-ob", type=int, default=16, help="Optimal batch size for inference")
    parser.add_argument("--minBatchSize", "-mb", type=int, default=1, help="Minimum batch size for inference")
    parser.add_argument("--fp16", default=True, action="store_true", help="Enable FP16 precision mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    main(
        onnx_path=args.onnx,
        engine_path=args.saveEngine,
        max_batchsize=args.maxBatchSize,
        opt_batchsize=args.optBatchSize,
        min_batchsize=args.minBatchSize,
        input_size=args.inputSize,
        use_fp16=args.fp16,
        verbose=args.verbose
    )
