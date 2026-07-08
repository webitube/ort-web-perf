"""
CLI test program that mirrors the web-txt2img engine pipeline to validate
a converted ONNX model's ability to load and run inference.

Pipeline (matches sd-turbo.ts adapter):
  tokenize → text_encoder → unet (1-step) → vae_decoder → PNG

Usage:
    conda activate ort-web-perf
    python test-model-cli.py ^
        --model-path sd-turbo-ort-web ^
        --prompt "a photo of an astronaut riding a horse" ^
        --output test-output.png

    # Test with custom model
    python test-model-cli.py ^
        --model-path mangledMerge-onnx ^
        --prompt "a castle in the clouds" ^
        --output castle.png
"""

import argparse
import os
import sys
import time
import math
import struct
import zlib

import numpy as np
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser(
        description="Test ONNX model loading and inference (mirrors web-txt2img pipeline)"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to the ONNX model directory (e.g., sd-turbo-ort-web)"
    )
    parser.add_argument(
        "--prompt", default="a photo of an astronaut riding a horse",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--output", default="test-output.png",
        help="Output PNG file path"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for latent noise generation"
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Output image width (default: 512)"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Output image height (default: 512)"
    )
    parser.add_argument(
        "--provider", default="auto",
        choices=["auto", "cpu", "cuda", "tensorrt"],
        help="Execution provider (default: auto = TensorRT+CUDA+CPU fallback)"
    )
    return parser.parse_args()


# ─── CLIP Tokenizer ───────────────────────────────────────────────────────

def tokenize_prompt(tokenizer_dir, prompt, max_length=77):
    """
    Tokenize using the real CLIPTokenizer from transformers.
    This matches what web-txt2img does (it uses the same tokenizer files).
    """
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )
    return encoded.input_ids[0].tolist()


# ─── Latent noise generation ──────────────────────────────────────────────

def mulberry32(seed):
    """Simple PRNG matching the JS mulberry32 in sd-turbo.ts."""
    state = seed & 0xFFFFFFFF

    def next():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t ^ (t + ((t ^ (t >> 7)) * 61 & 0xFFFFFFFF))) & 0xFFFFFFFF
        t = (t ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296.0

    return next


def randn_latents(shape, sigma, seed):
    """Generate random latents scaled by sigma, matching web engine randn_latents(shape, sigma, seed)."""
    rand = mulberry32(seed)
    size = 1
    for s in shape:
        size *= s

    values = []
    for _ in range(size):
        u = rand()
        v = rand()
        # Box-Muller transform
        while u == 0:
            u = rand()
        value = math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)
        values.append(value)

    arr = np.array(values, dtype=np.float32).reshape(shape)
    return arr * sigma


# ─── Scheduler helpers (matching sd-turbo.ts) ─────────────────────────────

def scale_model_inputs(latent, sigma):
    """Scale latent inputs for the UNet, matching EulerDiscreteScheduler.scale_model_input().
    
    Scales by (sigma**2 + 1)**0.5 to match the Euler algorithm.
    """
    data = np.asarray(latent)
    return data / ((sigma**2 + 1) ** 0.5)


def scheduler_step(out_sample, latent, sigma, sigma_next):
    """Single scheduler step for SD-Turbo (epsilon prediction), matching web engine eulerStep().
    
    Uses the ORIGINAL (unscaled) latent, NOT the scaled model input.
    Formula: x_{t-1} = x_t + epsilon * (sigma_{t-1} - sigma_t)
    where epsilon is the UNet output and x_t is the current latent.
    """
    out_data = np.asarray(out_sample)
    lat_data = np.asarray(latent)

    epsilon = out_data
    dt = sigma_next - sigma
    new_latents = lat_data + epsilon * dt

    return new_latents


# ─── PNG encoder (minimal, no PIL dependency) ─────────────────────────────

def tensor_to_png(tensor_data, width, height):
    """Convert float tensor [3, H, W] to PNG bytes."""
    # Tensor is [3, H, W] in NCHW format
    if tensor_data.ndim == 4 and tensor_data.shape[0] == 1:
        tensor_data = tensor_data[0]  # Remove batch dim

    # Convert from [C, H, W] to [H, W, C]
    image = np.transpose(tensor_data, (1, 2, 0))

    # VAE output is in [-1, 1] range, normalize to [0, 1] matching web engine: v / 2 + 0.5
    image = image / 2.0 + 0.5
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)

    # Encode as PNG
    return encode_png(image, width, height)


def encode_png(image, width, height):
    """Minimal PNG encoder."""
    def png_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b'IHDR', ihdr_data)

    # IDAT (raw image data with filter bytes)
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'  # Filter: None
        raw_data += image[y].tobytes()

    compressed = zlib.compress(raw_data)
    idat = png_chunk(b'IDAT', compressed)

    # IEND
    iend = png_chunk(b'IEND', b'')

    return signature + ihdr + idat + iend


# ─── Main test pipeline ───────────────────────────────────────────────────

def run_test(args):
    model_path = args.model_path
    prompt = args.prompt
    output_path = args.output
    seed = args.seed
    width = args.width
    height = args.height

    # Validate model structure
    required_files = [
        "unet/model.onnx",
        "text_encoder/model.onnx",
        "vae_decoder/model.onnx",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
    ]

    print("=" * 60)
    print("web-txt2img Model Validation Test")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Prompt:     {prompt}")
    print(f"Seed:       {seed}")
    print(f"Size:       {width}x{height}")
    print()

    # Check required files
    print("[1/6] Checking model structure...")
    missing = []
    for f in required_files:
        full_path = os.path.join(model_path, f)
        if not os.path.exists(full_path):
            missing.append(f)
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.1f} MB)")

    if missing:
        print(f"\n  ✗ Missing files: {', '.join(missing)}")
        print("  The model directory does not match the expected structure.")
        
        # Check if there are .safetensors files instead
        safetensors_files = []
        for item in os.listdir(model_path):
            if item.endswith('.safetensors'):
                size_gb = os.path.getsize(os.path.join(model_path, item)) / (1024**3)
                safetensors_files.append((item, size_gb))
        
        if safetensors_files:
            print("\n  💡 Found .safetensors files instead of ONNX models:")
            for name, size in safetensors_files:
                print(f"     - {name} ({size:.2f} GB)")
            print("\n  .safetensors files need to be converted to ONNX format first.")
            print("  Use the diffusers library or ONNX export tools to convert.")
            print("  Example conversion command:")
            print("    python -c \"from diffusers import StableDiffusionPipeline; \\\\")
            print("              pipe = StableDiffusionPipeline.from_pretrained('path/to/model'); \\\\")
            print("              pipe.save_pretrained('path/to/onnx/model', safe_serialization=False)\"")
        
        return False

    # Load tokenizer
    print("\n[2/6] Loading tokenizer...")
    tokenizer_dir = os.path.join(model_path, "tokenizer")
    tokens = tokenize_prompt(tokenizer_dir, prompt)
    print(f"  Tokenized: {len(tokens)} tokens")
    print(f"  First 10:  {tokens[:10]}")

    # Load ONNX sessions
    print("\n[3/6] Loading ONNX sessions...")

    # Build provider chain based on --provider flag
    # Check which providers are actually available
    available_providers = ort.get_available_providers()
    print(f"  Available providers: {', '.join(available_providers)}")

    if args.provider == "cpu":
        providers = [("CPUExecutionProvider", {})]
    elif args.provider == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            })
        ]
    else:  # "auto" or "tensorrt"
        # Use CUDA only — no CPU fallback.
        # FP16 models will fail on CPU, and ORT's fallback behavior
        # (reloading the session on CPU after a CUDA runtime error)
        # produces confusing errors. Better to fail fast on CUDA.
        if "CUDAExecutionProvider" in available_providers:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                })
            ]
        else:
            providers = [("CPUExecutionProvider", {})]

    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    # Disable graph optimizations that can cause issues with pre-optimized models
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Disable specific problematic fusions
    sess_options.add_session_config_entry("session.disable_prepacking", "1")
    sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    sess_options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
    sess_options.add_session_config_entry("session.use_ort_model_bytes_for_initializers", "1")

    start = time.time()
    text_encoder = ort.InferenceSession(
        os.path.join(model_path, "text_encoder", "model.onnx"),
        sess_options=sess_options,
        providers=providers,
    )
    print(f"  ✓ text_encoder loaded ({time.time() - start:.1f}s) [provider: {text_encoder.get_providers()[0]}]")

    start = time.time()
    unet = ort.InferenceSession(
        os.path.join(model_path, "unet", "model.onnx"),
        sess_options=sess_options,
        providers=providers,
    )
    print(f"  ✓ unet loaded ({time.time() - start:.1f}s) [provider: {unet.get_providers()[0]}]")

    start = time.time()
    vae_decoder = ort.InferenceSession(
        os.path.join(model_path, "vae_decoder", "model.onnx"),
        sess_options=sess_options,
        providers=providers,
    )
    print(f"  ✓ vae_decoder loaded ({time.time() - start:.1f}s) [provider: {vae_decoder.get_providers()[0]}]")

    # Run text encoder
    print("\n[4/6] Running text encoder...")
    input_ids = np.array([tokens], dtype=np.int32)
    start = time.time()
    enc_outputs = text_encoder.run(None, {"input_ids": input_ids})
    last_hidden_state = enc_outputs[0]
    print(f"  ✓ Encoded in {time.time() - start:.1f}s")
    print(f"    Output shape: {last_hidden_state.shape}")

    # Generate latents
    print("\n[5/6] Running UNet (1-step denoising)...")
    latent_shape = [1, 4, 64, 64]
    sigma = 14.6146
    vae_scaling_factor = 0.18215

    latent = randn_latents(latent_shape, sigma, seed)
    latent_model_input = scale_model_inputs(latent, sigma)

    timestep = np.array([999], dtype=np.int64)

    feed = {
        "sample": latent_model_input,
        "timestep": timestep,
        "encoder_hidden_states": last_hidden_state,
    }

    start = time.time()
    unet_outputs = unet.run(None, feed)
    out_sample = unet_outputs[0]
    print(f"  ✓ UNet inference in {time.time() - start:.1f}s")
    print(f"    Output shape: {out_sample.shape}")

    # Scheduler step (SD-Turbo: epsilon prediction, no CFG, 1 step)
    # sigma_next = 0.0 (final step, target is fully denoised)
    # IMPORTANT: use original latent (NOT scaled model input), matching web engine
    sigma_next = 0.0
    new_latents = scheduler_step(out_sample, latent, sigma, sigma_next)

    # Apply VAE scaling factor (matching web engine: scaleLatent divides by factor)
    vae_latent = new_latents / vae_scaling_factor

    # VAE decode
    print("\n[6/6] Running VAE decoder...")
    start = time.time()
    vae_outputs = vae_decoder.run(None, {"latent_sample": vae_latent})
    sample = vae_outputs[0]
    print(f"  ✓ Decoded in {time.time() - start:.1f}s")
    print(f"    Output shape: {sample.shape}")

    # Save PNG
    print(f"\nSaving image to {output_path}...")
    png_bytes = tensor_to_png(sample, width, height)
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"  ✓ Saved ({len(png_bytes)} bytes)")

    total_time = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"✓ Model validation PASSED")
    print(f"  Total inference time: {total_time:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")

    return True


def main():
    args = get_args()
    try:
        success = run_test(args)
        sys.exit(0 if success else 1)
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Test FAILED: {e}")
        
        # Provide diagnostic info for common ORT errors
        if "Type Error" in error_msg or "tensor(float16)" in error_msg:
            print("\n💡 DIAGNOSIS: FP16 precision mismatch detected.")
            print("  The model contains float16 (FP16) operations that require GPU or")
            print("  WebGPU backend. CPUExecutionProvider does not support FP16 tensors.")
            print("  Solutions:")
            print("    1. Use --provider CUDAExecutionProvider if you have an NVIDIA GPU")
            print("    2. Re-convert the model with float32 (FP32) precision")
            print("    3. Use onnxruntime-web with WebGPU backend in a browser")
        elif "SimplifiedLayerNormFusion" in error_msg or "GetIndexFromName" in error_msg:
            print("\n💡 DIAGNOSIS: ORT graph optimization conflict.")
            print("  The model has pre-optimized graph nodes that conflict with ORT's")
            print("  automatic optimization. Try disabling graph optimizations.")
        elif "Execution provider" in error_msg:
            print("\n💡 DIAGNOSIS: Execution provider not available.")
            print("  The specified provider may not be installed or compatible.")
            print("  Try --provider CPUExecutionProvider as fallback.")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
