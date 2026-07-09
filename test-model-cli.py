"""
CLI test program that mirrors the web-txt2img engine pipeline to validate
a converted ONNX model's ability to load and run inference.

Supports both SD-Turbo (Euler, 1-step) and SD 2.1 (DDIM, multi-step) models.
Auto-detects model type from scheduler config.

Pipeline:
  SD-Turbo: tokenize → text_encoder → unet (1-step Euler) → vae_decoder → PNG
  SD 2.1:   tokenize → text_encoder → unet (N-step DDIM) → vae_decoder → PNG

Usage:
    conda activate ort-web-perf
    python test-model-cli.py ^
        --model-path sd-turbo-ort-web ^
        --prompt "a photo of an astronaut riding a horse" ^
        --output test-output.png

    # Test with SD 2.1 model (auto-detects DDIM scheduler)
    python test-model-cli.py ^
        --model-path mangledMerge-onnx-fp16 ^
        --prompt "a castle in the clouds" ^
        --steps 20 ^
        --output castle.png
"""

import argparse
import json
import math
import os
import struct
import sys
import time
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
        "--steps", type=int, default=20,
        help="Number of denoising steps for DDIM scheduler (default: 20)"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=4.0,
        help="Classifier-Free Guidance scale for DDIM scheduler (default: 4.0). Set to 1.0 to disable CFG."
    )
    parser.add_argument(
        "--negative-prompt", default="",
        help="Negative prompt for CFG (default: empty string)"
    )
    parser.add_argument(
        "--scheduler", default="auto",
        choices=["auto", "ddim", "euler"],
        help="Scheduler to use (default: auto-detect from model config). Override to use Euler with DDIM models."
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


# ─── Model type detection ─────────────────────────────────────────────────

def detect_model_type(model_path):
    """Detect if the model is SD-Turbo (Euler, 1-step) or SD 2.1 (DDIM, multi-step).
    
    Reads scheduler config from model directory to determine the scheduler type.
    Returns a dict with model type info.
    """
    scheduler_config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
    
    if os.path.exists(scheduler_config_path):
        with open(scheduler_config_path, "r") as f:
            config = json.load(f)
        class_name = config.get("_class_name", "")
        prediction_type = config.get("prediction_type", "epsilon")
        
        if "Euler" in class_name:
            return {
                "type": "sd-turbo",
                "scheduler": "euler",
                "prediction_type": prediction_type,
                "steps": 1,
            }
        elif "DDIM" in class_name:
            return {
                "type": "sd-2.1",
                "scheduler": "ddim",
                "prediction_type": prediction_type,
                "steps": 20,
            }
    
    # Fallback: check model_index.json
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        with open(model_index_path, "r") as f:
            index = json.load(f)
        scheduler_entry = index.get("scheduler", [])
        if len(scheduler_entry) >= 2 and "Euler" in scheduler_entry[1]:
            return {"type": "sd-turbo", "scheduler": "euler", "prediction_type": "epsilon", "steps": 1}
        if len(scheduler_entry) >= 2 and "DDIM" in scheduler_entry[1]:
            return {"type": "sd-2.1", "scheduler": "ddim", "prediction_type": "v_prediction", "steps": 20}
    
    # Default fallback to SD-Turbo for backward compatibility
    print("  ⚠ Could not detect model type, defaulting to SD-Turbo (Euler)")
    return {"type": "sd-turbo", "scheduler": "euler", "prediction_type": "epsilon", "steps": 1}


# ─── Scheduler helpers ────────────────────────────────────────────────────

# Euler helpers (SD-Turbo, 1-step)
def scale_model_inputs(latent, sigma):
    """Scale latent inputs for the UNet, matching EulerDiscreteScheduler.scale_model_input()."""
    data = np.asarray(latent)
    return data / ((sigma**2 + 1) ** 0.5)


def euler_scheduler_step(out_sample, latent, sigma, sigma_next):
    """Single Euler scheduler step for SD-Turbo (epsilon prediction)."""
    epsilon = np.asarray(out_sample)
    lat_data = np.asarray(latent)
    dt = sigma_next - sigma
    return lat_data + epsilon * dt


# DDIM helpers (SD 2.1, multi-step)
def build_ddim_alphas_cumprod(beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000, beta_schedule="scaled_linear"):
    """Build alphas_cumprod matching DDIMScheduler."""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
    elif beta_schedule == "scaled_linear":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float64) ** 2
    else:
        raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas.astype(np.float32), alphas_cumprod.astype(np.float32)


def generate_ddim_timesteps(num_inference_steps, num_train_timesteps=1000, steps_offset=1):
    """Generate evenly-spaced timesteps matching DDIMScheduler.set_timesteps().
    
    Returns timesteps in DESCENDING order (high noise → low noise) for denoising.
    """
    step_ratio = num_train_timesteps // num_inference_steps
    timesteps = np.arange(0, num_inference_steps) * step_ratio + steps_offset
    timesteps = np.clip(timesteps, 0, num_train_timesteps - 1).astype(np.int64)
    # Reverse to go from high noise (start) to low noise (end)
    return timesteps[::-1]


def ddim_scheduler_step(noise, model_output, timestep, timestep_prev, alpha_prod_t, alpha_prod_t_prev, eta=0.0, prediction_type="epsilon"):
    """DDIM scheduler step, deterministic when eta=0.
    
    Matches DDIMScheduler.step() from diffusers.
    Supports both 'epsilon' and 'v_prediction' prediction types.
    
    Args:
        noise: current latent x_t
        model_output: raw UNet output (epsilon or v-prediction depending on type)
        timestep: current timestep t
        timestep_prev: previous timestep t-1 (not directly used, for logging)
        alpha_prod_t: alphas_cumprod[t]
        alpha_prod_t_prev: alphas_cumprod[t-1] (or 1.0 if t is the first step)
        eta: noise strength (0.0 = deterministic)
        prediction_type: 'epsilon' or 'v_prediction'
    """
    noise = np.asarray(noise, dtype=np.float32)
    model_output = np.asarray(model_output, dtype=np.float32)

    sqrt_alpha_prod = np.sqrt(alpha_prod_t)
    sqrt_beta_prod = np.sqrt(1.0 - alpha_prod_t)

    if prediction_type == "v_prediction":
        # v_prediction: model predicts v = alpha*x - sqrt(beta)*epsilon
        # Recover x_0 and epsilon from v (matches diffusers exactly)
        pred_original_sample = sqrt_alpha_prod * noise - sqrt_beta_prod * model_output
        pred_epsilon = sqrt_alpha_prod * model_output + sqrt_beta_prod * noise
    else:
        # epsilon prediction (default)
        pred_epsilon = model_output
        pred_original_sample = (noise - sqrt_beta_prod * pred_epsilon) / sqrt_alpha_prod

    # Compute previous sample
    sqrt_alpha_prod_prev = np.sqrt(alpha_prod_t_prev)
    sqrt_beta_prod_prev = np.sqrt(1.0 - alpha_prod_t_prev)

    if eta > 0.0:
        # Stochastic: add variance noise
        std_dev_t = eta * np.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev))
        pred_sample_direction = np.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon
        prev_sample = sqrt_alpha_prod_prev * pred_original_sample + pred_sample_direction + std_dev_t * noise
    else:
        # Deterministic (no variance)
        prev_sample = sqrt_alpha_prod_prev * pred_original_sample + sqrt_beta_prod_prev * pred_epsilon

    return prev_sample


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
    
    # Tokenize negative prompt for CFG
    negative_tokens = tokenize_prompt(tokenizer_dir, args.negative_prompt) if args.negative_prompt else None
    if negative_tokens is not None:
        print(f"  Negative prompt tokenized: {len(negative_tokens)} tokens")

    # Load ONNX sessions
    print("\n[3/6] Loading ONNX sessions...")

    # Build provider chain based on --provider flag
    # Check which providers are actually available
    ort.preload_dlls()
    available_providers = ort.get_available_providers()
    print(f"  Available providers: {', '.join(available_providers)}, args.provider={args.provider}")

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

    # Level 1 (Basic) stops complex transformer/attention fusion that causes the cast bug
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    # Disable graph optimizations that can cause issues with pre-optimized models
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Disable specific problematic fusions
    sess_options.add_session_config_entry("session.disable_prepacking", "1")
    sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    sess_options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
    sess_options.add_session_config_entry("session.use_ort_model_bytes_for_initializers", "1")

    # session = ort.InferenceSession("model.onnx", options=options, providers=['CUDAExecutionProvider'])

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
    
    # Detect expected dtype from model inputs (e.g. "tensor(int64)" -> np.int64)
    input_info = {inp.name: inp for inp in text_encoder.get_inputs()}
    ids_type_str = input_info["input_ids"].type if "input_ids" in input_info else "tensor(int32)"
    ids_dtype = np.int64 if "int64" in ids_type_str else np.int32
    input_ids = np.array([tokens], dtype=ids_dtype)
    
    # Build input feed - some models require attention_mask
    text_encoder_inputs = {"input_ids": input_ids}
    for input_name in input_info:
        if input_name != "input_ids":
            text_encoder_inputs[input_name] = np.ones_like(input_ids)
    
    start = time.time()
    enc_outputs = text_encoder.run(None, text_encoder_inputs)
    last_hidden_state = enc_outputs[0]
    print(f"  ✓ Encoded in {time.time() - start:.1f}s")
    print(f"    Output shape: {last_hidden_state.shape}")

    # Detect model type (SD-Turbo vs SD 2.1)
    model_info = detect_model_type(model_path)
    
    # Allow scheduler override
    scheduler = args.scheduler if args.scheduler != "auto" else model_info["scheduler"]
    
    print(f"\n[4.5/6] Detected model type: {model_info['type']} (detected: {model_info['scheduler']}, using: {scheduler})")
    print(f"  Prediction type: {model_info['prediction_type']}")
    
    # Use user-specified steps if provided, otherwise use model default
    num_inference_steps = args.steps if args.steps > 1 else model_info["steps"]
    if model_info["scheduler"] == "euler" and scheduler == "euler":
        num_inference_steps = 1  # SD-Turbo is always 1-step
    
    # Generate latents
    print(f"\n[5/6] Running UNet ({scheduler.upper()} {num_inference_steps}-step denoising)...")
    latent_shape = [1, 4, 64, 64]
    vae_scaling_factor = 0.18215
    
    # Build UNet input feed template
    unet_input_map = {inp.name: inp for inp in unet.get_inputs()}
    
    if scheduler == "euler":
        # ─── Euler denoising ──────────────────────────────────────────
        # SD-Turbo uses 1-step, SD 2.1 can use multi-step Euler
        is_sd_turbo = (model_info["scheduler"] == "euler")
        
        sigma = 14.6146
        latent = randn_latents(latent_shape, sigma, seed)
        
        # Encode negative prompt for CFG (SD 2.1 with Euler)
        guidance_scale = args.guidance_scale
        if negative_tokens is not None and not is_sd_turbo:
            neg_input_ids = np.array([negative_tokens], dtype=ids_dtype)
            neg_encoder_inputs = {"input_ids": neg_input_ids}
            for input_name in input_info:
                if input_name != "input_ids":
                    neg_encoder_inputs[input_name] = np.ones_like(neg_input_ids)
            neg_hidden_state = text_encoder.run(None, neg_encoder_inputs)[0]
        elif not is_sd_turbo:
            neg_hidden_state = np.zeros_like(last_hidden_state)
        
        if not is_sd_turbo and guidance_scale > 1.0:
            print(f"  ✓ CFG enabled (guidance_scale={guidance_scale})")
        
        # Euler timesteps for multi-step
        if is_sd_turbo:
            # SD-Turbo: single step at t=999
            sigmas = [sigma, 0.0]
        else:
            # SD 2.1 Euler: generate sigmas from timesteps
            timesteps = generate_ddim_timesteps(num_inference_steps, num_train_timesteps=1000, steps_offset=1)
            # Convert timesteps to sigmas: sigma_t = sqrt((1-alpha_t)/alpha_t)
            alphas, alphas_cumprod = build_ddim_alphas_cumprod(
                beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000, beta_schedule="scaled_linear"
            )
            sigmas = []
            for t in timesteps:
                alpha = float(alphas_cumprod[int(t)])
                sigmas.append(np.sqrt((1.0 - alpha) / alpha))
            sigmas.append(0.0)  # Final sigma = 0
        
        total_unet_time = 0.0
        for i in range(len(sigmas) - 1):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            latent_model_input = scale_model_inputs(latent, sigma_curr)
            timestep = np.array([999 if is_sd_turbo else int(timesteps[i])], dtype=np.int64)
            
            def run_unet(encoder_hidden, latent_input):
                feed = {}
                for name, inp in unet_input_map.items():
                    if "sample" in name and "hidden" not in name:
                        feed[name] = latent_input
                    elif "timestep" in name:
                        feed[name] = timestep
                    elif "encoder" in name or "hidden" in name:
                        feed[name] = encoder_hidden
                    else:
                        shape = [1 if (s is None or isinstance(s, str)) else s for s in inp.shape]
                        feed[name] = np.zeros(shape, dtype=np.float32)
                return unet.run(None, feed)[0]
            
            step_start = time.time()
            
            cond_output = run_unet(last_hidden_state, latent_model_input)
            
            if not is_sd_turbo and guidance_scale > 1.0:
                uncond_output = run_unet(neg_hidden_state, latent_model_input)
                out_sample = uncond_output + guidance_scale * (cond_output - uncond_output)
            else:
                out_sample = cond_output
            
            step_time = time.time() - step_start
            total_unet_time += step_time
            
            # Euler scheduler step
            new_latents = euler_scheduler_step(out_sample, latent, sigma_curr, sigma_next)
            latent = new_latents
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  ✓ Step {i+1}/{len(sigmas)-1} (sigma={sigma_curr:.2f}) in {step_time:.2f}s "
                      f"[latent: mean={latent.mean():.4f}, std={latent.std():.4f}]")
        
        print(f"  ✓ UNet denoising complete in {total_unet_time:.1f}s ({len(sigmas)-1} steps)")
        print(f"    Output shape: {latent.shape}")
        
    else:
        # ─── SD 2.1: DDIM multi-step denoising with CFG ───────────────
        alphas, alphas_cumprod = build_ddim_alphas_cumprod(
            beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000, beta_schedule="scaled_linear"
        )
        timesteps = generate_ddim_timesteps(num_inference_steps, num_train_timesteps=1000, steps_offset=1)
        
        # Encode negative prompt for CFG
        guidance_scale = args.guidance_scale
        if negative_tokens is not None:
            neg_input_ids = np.array([negative_tokens], dtype=ids_dtype)
            neg_encoder_inputs = {"input_ids": neg_input_ids}
            for input_name in input_info:
                if input_name != "input_ids":
                    neg_encoder_inputs[input_name] = np.ones_like(neg_input_ids)
            neg_hidden_state = text_encoder.run(None, neg_encoder_inputs)[0]
        else:
            # Empty negative prompt: zeros
            neg_hidden_state = np.zeros_like(last_hidden_state)
        
        if guidance_scale > 1.0:
            print(f"  ✓ CFG enabled (guidance_scale={guidance_scale})")
        else:
            print(f"  ✓ CFG disabled (guidance_scale={guidance_scale})")
        
        # Initial pure Gaussian noise
        latent = randn_latents(latent_shape, 1.0, seed)
        
        total_unet_time = 0.0
        for i, t in enumerate(timesteps):
            t_scalar = np.int64(t)
            # Next timestep (lower, since timesteps are descending)
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
            
            alpha_prod_t = float(alphas_cumprod[t_scalar])
            alpha_prod_t_prev = float(alphas_cumprod[t_next])
            
            def run_unet(encoder_hidden):
                feed = {}
                for name, inp in unet_input_map.items():
                    if "sample" in name and "hidden" not in name:
                        feed[name] = latent
                    elif "timestep" in name:
                        feed[name] = np.array([t_scalar], dtype=np.int64)
                    elif "encoder" in name or "hidden" in name:
                        feed[name] = encoder_hidden
                    else:
                        shape = [1 if (s is None or isinstance(s, str)) else s for s in inp.shape]
                        feed[name] = np.zeros(shape, dtype=np.float32)
                return unet.run(None, feed)[0]
            
            step_start = time.time()
            
            # Run UNet with conditioned (positive) prompt
            cond_output = run_unet(last_hidden_state)
            
            if guidance_scale > 1.0:
                # Run UNet with unconditioned (negative) prompt
                uncond_output = run_unet(neg_hidden_state)
                # CFG: uncond + scale * (cond - uncond)
                out_sample = uncond_output + guidance_scale * (cond_output - uncond_output)
            else:
                out_sample = cond_output
            
            step_time = time.time() - step_start
            total_unet_time += step_time
            
            # Debug: check UNet output
            if i == 0:
                print(f"    [DEBUG] UNet cond output: mean={cond_output.mean():.4f}, std={cond_output.std():.4f}")
                if guidance_scale > 1.0:
                    print(f"    [DEBUG] UNet uncond output: mean={uncond_output.mean():.4f}, std={uncond_output.std():.4f}")
                    print(f"    [DEBUG] CFG combined output: mean={out_sample.mean():.4f}, std={out_sample.std():.4f}")
                print(f"    [DEBUG] alpha_prod_t={alpha_prod_t:.6f}, alpha_prod_t_prev={alpha_prod_t_prev:.6f}")
            
            # DDIM scheduler step (pass prediction_type for v_prediction support)
            latent = ddim_scheduler_step(
                latent, out_sample, t_scalar, t_next,
                alpha_prod_t, alpha_prod_t_prev, eta=0.0,
                prediction_type=model_info["prediction_type"]
            )
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  ✓ Step {i+1}/{num_inference_steps} (t={t_scalar}) in {step_time:.2f}s "
                      f"[latent: mean={latent.mean():.4f}, std={latent.std():.4f}]")
        
        print(f"  ✓ UNet denoising complete in {total_unet_time:.1f}s ({num_inference_steps} steps)")
        print(f"    Output shape: {latent.shape}")
    
    # Apply VAE scaling factor (matching web engine: scaleLatent divides by factor)
    vae_latent = latent / vae_scaling_factor

    # VAE decode
    print("\n[6/6] Running VAE decoder...")
    start = time.time()
    vae_input_name = vae_decoder.get_inputs()[0].name
    vae_outputs = vae_decoder.run(None, {vae_input_name: vae_latent})
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
