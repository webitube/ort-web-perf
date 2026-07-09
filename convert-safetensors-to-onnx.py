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
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import onnx


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

    # Try from_single_file first, fall back to direct loading
    kwargs = {}
    if original_config_file:
        kwargs["original_config_file"] = original_config_file

    print(f"  Loading from: {input_path}")
    try:
        pipeline = StableDiffusionPipeline.from_single_file(
            input_path,
            **kwargs,
        )
    except Exception as e:
        print(f"  from_single_file failed: {type(e).__name__}: {e}")
        print("  Falling back to direct safetensors loading...")
        pipeline = load_safetensors_directly(input_path, original_config_file)

    # Save safety_checker and feature_extractor as None to match sd-turbo-ort-web
    pipeline.safety_checker = None
    pipeline.feature_extractor = None

    print(f"  Saving to: {output_diffusers_dir}")
    pipeline.save_pretrained(output_diffusers_dir, safe_serialization=False)
    print(f"  Saved diffusers pipeline to: {output_diffusers_dir}")
    return output_diffusers_dir


def load_safetensors_directly(input_path, original_config_file):
    """Directly load safetensors checkpoint components into a diffusers pipeline.
    
    This bypasses HuggingFace Hub entirely by loading components manually
    from the checkpoint using the original SD config YAML.
    """
    import torch
    import yaml
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTextConfig

    # Load safetensors state dict
    print("  Loading safetensors state dict...")
    import safetensors.torch
    state_dict = safetensors.torch.load_file(input_path)

    # Determine model architecture
    if original_config_file:
        print(f"  Using config: {original_config_file}")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        original_config_file = os.path.join(script_dir, "sd21-inference.yaml")
        print(f"  Using default config: {original_config_file}")

    with open(original_config_file, "r") as f:
        config = yaml.safe_load(f)

    unet_params = config["model"]["unet_config"]["params"]
    context_dim = unet_params.get("context_dim", 1024)
    is_sd2 = context_dim == 1024

    print(f"  Detected: {'SD 2.x' if is_sd2 else 'SD 1.x'} (context_dim={context_dim})")

    # --- Load Text Encoder (OpenCLIP for SD 2.x) ---
    print("  Loading text encoder...")
    text_config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1024 if is_sd2 else 768,
        intermediate_size=4096 if is_sd2 else 3072,
        num_hidden_layers=24 if is_sd2 else 12,
        num_attention_heads=16,
        max_position_embeddings=77,
        hidden_act="gelu",
        projection_dim=768,
    )
    text_encoder = CLIPTextModel(text_config)

    # Map checkpoint keys to diffusers keys for OpenCLIP
    te_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("cond_stage_model.model."):
            continue
        new_key = key.replace("cond_stage_model.model.", "text_model.")
        # Handle OpenCLIP key mapping
        if new_key.startswith("text_model.transformer."):
            new_key = new_key.replace("text_model.transformer.", "text_model.encoder.")
        if new_key.endswith(".attn.in_proj_weight"):
            # Split combined attention weights
            continue
        if new_key.endswith(".attn.in_proj_bias"):
            continue
        te_state_dict[new_key] = value

    # Handle token embedding
    if "cond_stage_model.model.token_embedding.weight" in state_dict:
        te_state_dict["text_model.embeddings.token_embedding.weight"] = \
            state_dict["cond_stage_model.model.token_embedding.weight"]
    # Handle positional embedding
    if "cond_stage_model.model.positional_embedding" in state_dict:
        te_state_dict["text_model.embeddings.position_ids"] = \
            state_dict["cond_stage_model.model.positional_embedding"]
    # Handle layer norm
    if "cond_stage_model.model.ln_final.bias" in state_dict:
        te_state_dict["text_model.final_layer_norm.bias"] = \
            state_dict["cond_stage_model.model.ln_final.bias"]
    if "cond_stage_model.model.ln_final.weight" in state_dict:
        te_state_dict["text_model.final_layer_norm.weight"] = \
            state_dict["cond_stage_model.model.ln_final.weight"]

    text_encoder.load_state_dict(te_state_dict, strict=False)
    text_encoder.eval()

    # --- Load UNet ---
    print("  Loading UNet...")
    unet_config = {
        "act_fn": "silu",
        "attention_head_dim": unet_params.get("num_head_channels", 64) // 8 or 8,
        "block_out_channels": [
            unet_params["model_channels"] * m
            for m in unet_params["channel_mult"]
        ],
        "center_input_sample": False,
        "cross_attention_dim": context_dim,
        "down_block_types": [
            "DownBlock2D" if i == 0 else "CrossAttnDownBlock2D"
            for i in range(len(unet_params["channel_mult"]))
        ],
        "dual_cross_attention": False,
        "in_channels": unet_params.get("in_channels", 4),
        "layers_per_block": unet_params.get("num_res_blocks", 2),
        "mid_block_scale_factor": 1,
        "norm_num_groups": 32,
        "out_channels": unet_params.get("out_channels", 4),
        "sample_size": 64,
        "up_block_types": [
            "CrossAttnUpBlock2D" if i < 3 else "UpBlock2D"
            for i in range(len(unet_params["channel_mult"]))
        ],
    }
    unet = UNet2DConditionModel.from_config(unet_config)

    # Map checkpoint keys to diffusers UNet keys
    unet_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model."):
            new_key = key.replace("model.diffusion_model.", "")
            unet_state_dict[new_key] = value
    unet.load_state_dict(unet_state_dict, strict=False)
    unet.eval()
    unet.set_attn_processor(AttnProcessor2_0())

    # --- Load VAE ---
    print("  Loading VAE...")
    vae_config = {
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        "latent_channels": 4,
        "layers_per_block": 2,
        "act_fn": "silu",
        "sample_size": 512,
        "in_channels": 3,
        "out_channels": 3,
    }
    vae = AutoencoderKL.from_config(vae_config)

    vae_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.first_stage_model."):
            new_key = key.replace("model.first_stage_model.", "")
            vae_state_dict[new_key] = value
    vae.load_state_dict(vae_state_dict, strict=False)
    vae.eval()

    # --- Scheduler ---
    print("  Creating scheduler...")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        trained_betas=None,
        clip_sample=False,
        steps_offset=1,
        prediction_type="v_prediction",
    )

    # --- Tokenizer ---
    print("  Creating tokenizer...")
    from transformers import CLIPTokenizer
    tokenizer_name = "openai/clip-vit-large-patch14" if is_sd2 else "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

    # --- Build Pipeline ---
    print("  Building pipeline...")
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    return pipeline


def step2_export_onnx(diffusers_dir, output_onnx_dir):
    """Export diffusers pipeline to ONNX using torch.onnx.export directly."""
    print("=" * 60)
    print("Step 2: Exporting to ONNX (fp16) with torch.onnx.export...")
    print("=" * 60)

    # Remove output dir if it exists
    if os.path.exists(output_onnx_dir):
        shutil.rmtree(output_onnx_dir)

    import torch
    from diffusers import StableDiffusionPipeline

    # Load the pipeline
    # Note: safety_checker, feature_extractor, and tokenizer are intentionally None
    # to match the pipeline saved in Step 1
    print(f"  Loading pipeline from: {diffusers_dir}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        diffusers_dir,
        safety_checker=None,
        feature_extractor=None,
        tokenizer=None,
        requires_safety_checker=False,
    )

    # Export UNet
    print("  Exporting UNet...")
    unet_dir = os.path.join(output_onnx_dir, "unet")
    os.makedirs(unet_dir, exist_ok=True)
    unet = pipeline.unet.eval()
    unet.to("cpu")

    # Check if the model uses v_prediction (SD 2.x default)
    # If so, wrap the UNet to convert v → epsilon at export time
    # so the ONNX output matches what the web engine expects (epsilon).
    prediction_type = pipeline.scheduler.config.prediction_type
    print(f"  Scheduler prediction_type: {prediction_type}")

    if prediction_type == "v_prediction":
        print("  Wrapping UNet with v→epsilon conversion layer...")

        # Build alphas_cumprod table from scheduler betas
        betas = pipeline.scheduler.betas.numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        class VToEpsilonWrapper(torch.nn.Module):
            """Wraps a v-prediction UNet to output epsilon instead of v.

            Diffusers v_prediction formula (from DDIMScheduler.step):
                pred_epsilon = sqrt(alpha_prod_t) * model_output + sqrt(beta_prod_t) * sample
            
            Where:
                alpha_prod_t = alphas_cumprod[timestep]
                beta_prod_t  = 1 - alpha_prod_t
                model_output = v (UNet output)
                sample       = x_t (current latent)
            """
            def __init__(self, unet, alphas_cumprod):
                super().__init__()
                self.unet = unet
                # Store alphas_cumprod as a buffer (1000 entries, indices 0-999)
                # Timesteps passed to the UNet directly index into this array.
                self.register_buffer(
                    "alphas_cumprod",
                    torch.tensor(alphas_cumprod, dtype=torch.float32),
                )

            def forward(self, sample, timestep, encoder_hidden_states):
                v = self.unet(sample, timestep, encoder_hidden_states)[0]
                # Clamp timestep to valid range [0, 999]
                t = torch.clamp(timestep, 0, len(self.alphas_cumprod) - 1)
                alpha_prod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1).expand(-1, *sample.shape[1:])
                beta_prod_t = 1.0 - alpha_prod_t
                # diffusers v_prediction -> epsilon conversion
                epsilon = torch.sqrt(alpha_prod_t) * v + torch.sqrt(beta_prod_t) * sample
                return epsilon

        wrapped_unet = VToEpsilonWrapper(unet, alphas_cumprod).eval()
    else:
        wrapped_unet = unet

    # Create dummy inputs for UNet
    sample = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 77, 1024, dtype=torch.float32)

    unet_path = os.path.join(unet_dir, "model.onnx")
    torch.onnx.export(
        wrapped_unet,
        (sample, timestep, encoder_hidden_states),
        unet_path,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["sample"],
        dynamic_axes={
            "sample": [0, 2, 3],
            "timestep": [0],
            "encoder_hidden_states": [0, 1],
        },
        opset_version=18,
        do_constant_folding=True,
    )

    # Export VAE Encoder
    print("  Exporting VAE Encoder...")
    vae_enc_dir = os.path.join(output_onnx_dir, "vae_encoder")
    os.makedirs(vae_enc_dir, exist_ok=True)
    vae_enc = pipeline.vae.encoder.eval()
    vae_enc.to("cpu")
    
    sample = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    vae_enc_path = os.path.join(vae_enc_dir, "model.onnx")
    torch.onnx.export(
        vae_enc,
        (sample,),
        vae_enc_path,
        input_names=["sample"],
        output_names=["latent"],
        dynamic_axes={"sample": [0, 2, 3]},
        opset_version=18,
        do_constant_folding=True,
    )

    # Export VAE Decoder
    print("  Exporting VAE Decoder...")
    vae_dec_dir = os.path.join(output_onnx_dir, "vae_decoder")
    os.makedirs(vae_dec_dir, exist_ok=True)
    vae_dec = pipeline.vae.decoder.eval()
    vae_dec.to("cpu")
    
    latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    vae_dec_path = os.path.join(vae_dec_dir, "model.onnx")
    torch.onnx.export(
        vae_dec,
        (latent,),
        vae_dec_path,
        input_names=["latent"],
        output_names=["sample"],
        dynamic_axes={"latent": [0, 2, 3]},
        opset_version=18,
        do_constant_folding=True,
    )

    # Export Text Encoder
    print("  Exporting Text Encoder...")
    te_dir = os.path.join(output_onnx_dir, "text_encoder")
    os.makedirs(te_dir, exist_ok=True)
    te = pipeline.text_encoder.eval()
    te.to("cpu")
    
    input_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    te_path = os.path.join(te_dir, "model.onnx")
    torch.onnx.export(
        te,
        (input_ids, attention_mask),
        te_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": [0, 1],
            "attention_mask": [0, 1],
        },
        opset_version=18,
        do_constant_folding=True,
    )

    # Copy model_index.json so the optimizer can find it
    src_index = os.path.join(diffusers_dir, "model_index.json")
    dst_index = os.path.join(output_onnx_dir, "model_index.json")
    if os.path.exists(src_index):
        shutil.copy2(src_index, dst_index)

    print(f"  ONNX models exported to: {output_onnx_dir}")
    return output_onnx_dir


def step3_optimize_onnx(output_onnx_dir):
    """Optimize ONNX models using onnxruntime.transformers."""
    print("=" * 60)
    print("Step 3: Optimizing ONNX models...")
    print("=" * 60)

    # Use a temp dir for optimization output so we don't lose the originals on failure
    opt_output_dir = output_onnx_dir + "-opt-tmp"

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
        "-o", opt_output_dir,
        "--overwrite",
        "--float16",
    ] + opt.split()

    print(f"  Running optimize_pipeline...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        print("  WARNING: Optimization failed, continuing with unoptimized models.")
        # Clean up temp dir on failure
        if os.path.exists(opt_output_dir):
            shutil.rmtree(opt_output_dir)
    else:
        print("  Optimization complete.")
        # Replace originals with optimized models
        shutil.rmtree(output_onnx_dir)
        os.rename(opt_output_dir, output_onnx_dir)
        # Merge external .onnx.data back into self-contained .onnx files
        for model_file in glob.glob(os.path.join(output_onnx_dir, "**", "model.onnx"), recursive=True):
            data_file = model_file + ".data"
            if os.path.exists(data_file):
                print(f"  Merging external data: {os.path.relpath(model_file, output_onnx_dir)}")
                model = onnx.load(model_file)
                onnx.save(model, model_file, save_as_external_data=False)
                os.remove(data_file)
                print(f"    Removed: {os.path.relpath(data_file, output_onnx_dir)}")


def step4_copy_metadata(diffusers_dir, output_onnx_dir):
    """Copy metadata files (config.json, tokenizer, scheduler, etc.) from diffusers dir to ONNX output."""
    print("=" * 60)
    print("Step 4: Copying metadata files...")
    print("=" * 60)

    # Copy per-component config.json files
    for component in ["text_encoder", "unet", "vae_decoder", "vae_encoder"]:
        src_config = os.path.join(diffusers_dir, component, "config.json")
        dst_dir = os.path.join(output_onnx_dir, component)
        if os.path.exists(src_config):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_config, os.path.join(dst_dir, "config.json"))
            print(f"  Copied: {component}/config.json")

    # Copy scheduler config
    src_scheduler = os.path.join(diffusers_dir, "scheduler")
    dst_scheduler = os.path.join(output_onnx_dir, "scheduler")
    if os.path.isdir(src_scheduler):
        shutil.copytree(src_scheduler, dst_scheduler, dirs_exist_ok=True)
        print(f"  Copied: scheduler/")

    # Copy tokenizer files
    src_tokenizer = os.path.join(diffusers_dir, "tokenizer")
    dst_tokenizer = os.path.join(output_onnx_dir, "tokenizer")
    if os.path.isdir(src_tokenizer):
        shutil.copytree(src_tokenizer, dst_tokenizer, dirs_exist_ok=True)
        print(f"  Copied: tokenizer/")

    # Copy model_index.json (may already exist, ensure it's up to date)
    src_index = os.path.join(diffusers_dir, "model_index.json")
    if os.path.exists(src_index):
        shutil.copy2(src_index, os.path.join(output_onnx_dir, "model_index.json"))
        print(f"  Copied: model_index.json")

    # Copy any other top-level files (e.g., README, LICENSE)
    for fname in os.listdir(diffusers_dir):
        fpath = os.path.join(diffusers_dir, fname)
        if os.path.isfile(fpath) and fname.endswith((".md", ".txt", ".py")):
            shutil.copy2(fpath, os.path.join(output_onnx_dir, fname))
            print(f"  Copied: {fname}")


def step5_wrap_fp16(output_onnx_dir, script_dir):
    """Wrap fp16 inputs/outputs with Cast nodes for ORT-Web compatibility."""
    print("=" * 60)
    print("Step 5: Wrapping fp16 I/O with Cast nodes...")
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

    diffusers_dir = args.output + "-diffusers-intermediate"
    onnx_dir = args.output

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Clean up any previous intermediate dirs
    if os.path.exists(diffusers_dir):
        print(f"Cleaning up previous intermediate directory: {diffusers_dir}")
        shutil.rmtree(diffusers_dir)

    # Step 1: Load safetensors -> diffusers
    step1_load_and_save_diffusers(
        args.input, diffusers_dir, args.original_config_file, args.device
    )

    # Step 2: diffusers -> ONNX (fp16)
    step2_export_onnx(diffusers_dir, onnx_dir)
    
    # Step 3: Optimize ONNX
    if not args.no_optimize:
        step3_optimize_onnx(onnx_dir)

    # Step 4: Copy metadata files (before cleanup)
    step4_copy_metadata(diffusers_dir, onnx_dir)

    # Step 5: Wrap fp16 I/O
    if not args.no_wrap_fp16:
        step5_wrap_fp16(onnx_dir, script_dir)

    # Cleanup intermediate dir
    if os.path.exists(diffusers_dir):
        print(f"\nCleaning up intermediate directory: {diffusers_dir}")
        shutil.rmtree(diffusers_dir)

    print("\n" + "=" * 60)
    print(f"Done! ONNX model saved to: {onnx_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
