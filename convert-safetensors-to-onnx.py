"""
Convert a .safetensors Stable Diffusion checkpoint (with unet, clip, vae)
into ONNX diffusers format (fp16), similar to sd-turbo-ort-web.

Usage:
    conda activate ort-web-perf
    python convert-safetensors-to-onnx.py ^
        --input mangledMerge\\mangledMerge_v3.safetensors ^
        --output mangledMerge-onnx

Requires: diffusers, transformers, optimum[onnxruntime], onnxruntime
"""

import argparse
import os
import shutil
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert safetensors SD checkpoint to ONNX diffusers format (fp16)"
    )
    parser.add_argument(
        "--input", required=True, help="Path to the .safetensors checkpoint file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for the ONNX model"
    )
    parser.add_argument(
        "--original-config-file",
        default=None,
        help="Path to original SD config YAML (e.g. v1-inference.yaml). "
        "Auto-detected if not provided.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip ONNX optimization step (optimum-cli)",
    )
    parser.add_argument(
        "--no-wrap-fp16",
        action="store_true",
        help="Skip fp16 wrapping of ONNX models",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for loading the model (default: cpu)",
    )
    return parser.parse_args()


def step1_load_and_save_diffusers(input_path, output_diffusers_dir, original_config_file, device):
    """Load safetensors checkpoint and save as diffusers pipeline."""
    print("=" * 60)
    print("Step 1: Loading safetensors and saving as diffusers pipeline...")
    print("=" * 60)

    import torch
    from diffusers import StableDiffusionPipeline

    kwargs = {}
    if original_config_file:
        kwargs["original_config_file"] = original_config_file

    pipeline = StableDiffusionPipeline.from_single_file(
        input_path,
        **kwargs,
    )

    # Save safety_checker and feature_extractor as None to match sd-turbo-ort-web
    pipeline.safety_checker = None
    pipeline.feature_extractor = None

    pipeline.save_pretrained(output_diffusers_dir, safe_serialization=False)
    print(f"  Saved diffusers pipeline to: {output_diffusers_dir}")
    return output_diffusers_dir


def step2_export_onnx(diffusers_dir, output_onnx_dir):
    """Export diffusers pipeline to ONNX using optimum-cli."""
    print("=" * 60)
    print("Step 2: Exporting to ONNX (fp16) with optimum-cli...")
    print("=" * 60)

    # Remove output dir if it exists
    if os.path.exists(output_onnx_dir):
        shutil.rmtree(output_onnx_dir)

    cmd = [
        "optimum-cli", "export", "onnx",
        "--fp16",
        diffusers_dir,
        output_onnx_dir,
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"optimum-cli export failed with code {result.returncode}")
    print(f"  ONNX models exported to: {output_onnx_dir}")
    return output_onnx_dir


def step3_optimize_onnx(output_onnx_dir):
    """Optimize ONNX models using onnxruntime.transformers."""
    print("=" * 60)
    print("Step 3: Optimizing ONNX models...")
    print("=" * 60)

    # Same options as sd-turbo-for-web.sh but without custom ops for web compatibility
    opt = (
        "--no_attention_mask "
        "--disable_skip_layer_norm "
        "--disable_attention "
        "--disable_nhwc_conv "
        "--disable_group_norm "
        "--disable_skip_group_norm "
        "--disable_embed_layer_norm "
        "--disable_bias_splitgelu "
        "--disable_bias_skip_layer_norm "
        "--disable_bias_gelu "
        "--disable_packed_kv "
        "--disable_packed_qkv"
    )

    cmd = [
        "python", "-m",
        "onnxruntime.transformers.models.stable_diffusion.optimize_pipeline",
        "-i", output_onnx_dir,
        "-o", output_onnx_dir,
        "--overwrite",
        "--float16",
    ] + opt.split()

    print(f"  Running optimize_pipeline...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        # Non-fatal - optimization may fail on some models
        print("  WARNING: Optimization failed, continuing with unoptimized models.")
    else:
        print("  Optimization complete.")


def step4_wrap_fp16(output_onnx_dir, script_dir):
    """Wrap fp16 inputs/outputs with Cast nodes for ORT-Web compatibility."""
    print("=" * 60)
    print("Step 4: Wrapping fp16 I/O with Cast nodes...")
    print("=" * 60)

    wrap_script = os.path.join(script_dir, "onnx-wrap-fp16.py")
    if not os.path.exists(wrap_script):
        print(f"  WARNING: onnx-wrap-fp16.py not found at {wrap_script}, skipping.")
        return

    for component in ["unet", "vae_decoder", "text_encoder"]:
        model_path = os.path.join(output_onnx_dir, component, "model.onnx")
        if os.path.exists(model_path):
            cmd = [
                "python", wrap_script,
                "--input", model_path,
                "--output", model_path,
            ]
            print(f"  Wrapping {component}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  STDERR: {result.stderr}")
            else:
                print(f"  Wrapped {component}.")
        else:
            print(f"  Skipping {component} (no model.onnx found)")


def main():
    args = get_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Create intermediate diffusers directory
    diffusers_dir = args.output + "-diffusers-intermediate"
    onnx_dir = args.output

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Load safetensors -> diffusers
    step1_load_and_save_diffusers(
        args.input, diffusers_dir, args.original_config_file, args.device
    )

    # Step 2: diffusers -> ONNX (fp16)
    step2_export_onnx(diffusers_dir, onnx_dir)

    # Step 3: Optimize ONNX
    if not args.no_optimize:
        step3_optimize_onnx(onnx_dir)

    # Step 4: Wrap fp16 I/O
    if not args.no_wrap_fp16:
        step4_wrap_fp16(onnx_dir, script_dir)

    # Cleanup intermediate dir
    if os.path.exists(diffusers_dir):
        print(f"\nCleaning up intermediate directory: {diffusers_dir}")
        shutil.rmtree(diffusers_dir)

    print("\n" + "=" * 60)
    print(f"Done! ONNX model saved to: {onnx_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
