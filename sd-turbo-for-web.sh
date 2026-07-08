#
# Script to create a Stable Diffusion Turbo ONNX model optimized for ONNX Runtime Web
# Prerequisites: pip install -U transformers diffusers optimum onnxruntime
#

root=d:/Dev/ort-web-perf
model=$root/onnx-sd-turbo              # Output dir for the default FP32 ONNX export
model_fp16=$root/onnx-sd-turbo-fp16   # Output dir for the FP16 ONNX export
out=$root/sd-opt                      # Final output dir after ORT optimization

org=stabilityai/sd-turbo              # The Hugging Face model to export

# ---------------------------------------------------------------------------
# Step 1: Export the model to ONNX (FP32) if not already done
# This is the baseline export used later by the ORT optimizer
# ---------------------------------------------------------------------------
if [ ! -d $model ] ; then
   optimum-cli export onnx -m $org $model
fi

# ---------------------------------------------------------------------------
# Step 2: Export the model to ONNX in FP16 (half-precision) if not already done
# --fp16:  export weights in float16 instead of float32 (smaller files, faster)
# --device cuda: run the export on GPU (required for FP16 support)
#
# After export, post-process each sub-model to clean up the graph and ensure
# compatibility with ONNX Runtime Web, which may not support native FP16 I/O.
# ---------------------------------------------------------------------------
if [ ! -d $model_fp16 ] ; then

  # --- Initial FP16 export ---
  optimum-cli export onnx --fp16 --device cuda -m $org $model_fp16

  # --- UNet (the core diffusion/denoising network) ---

  # Remove constant nodes that were fused or inlined unnecessarily,
  # simplifying the graph structure
  python onnx-remove-const.py --input $model_fp16/unet/model.onnx --output  $model_fp16/unet/model.onnx

  # Remove any FP64 (double precision) operations/tensors that may have
  # crept in during export, ensuring the graph stays pure FP16/FP32
  python onnx-remove-double.py --input $model_fp16/unet/model.onnx --output  $model_fp16/unet/model.onnx

  # Wrap FP16 inputs/outputs with Cast nodes so the external I/O interface
  # is FP32 while internal computation remains FP16. This is needed because
  # ONNX Runtime Web requires FP32 at the graph boundary.
  python onnx-wrap-fp16.py --input $model_fp16/unet/model.onnx --output  $model_fp16/unet/model.onnx

  # --- VAE Decoder (converts latent space back to pixel image) ---

  # Only needs the FP16 I/O wrapping; no const/double cleanup needed
  # (the VAE decoder graph is simpler and doesn't have the same issues)
  python onnx-wrap-fp16.py --input $model_fp16/vae_decoder/model.onnx --output  $model_fp16/vae_decoder/model.onnx

  # --- Text Encoder (CLIP: converts prompt text to embeddings) ---

  # Same cleanup pipeline as UNet: remove constants, remove doubles, wrap I/O
  python onnx-remove-const.py --input $model_fp16/text_encoder/model.onnx --output  $model_fp16/text_encoder/model.onnx
  python onnx-remove-double.py --input $model_fp16/text_encoder/model.onnx --output  $model_fp16/text_encoder/model.onnx
  python onnx-wrap-fp16.py --input $model_fp16/text_encoder/model.onnx --output  $model_fp16/text_encoder/model.onnx
fi

# ---------------------------------------------------------------------------
# Step 3: Run the ONNX Runtime optimizer on the FP32 export
#
# This applies graph-level optimizations specific to Stable Diffusion:
#   --disable_attention:          decompose custom Attention ops into primitives
#   --disable_skip_layer_norm:    decompose SkipLayerNorm into separate ops
#   --disable_nhwc_conv:          keep convolutions in NCHW format (Web-compatible)
#   --disable_group_norm:         decompose GroupNorm into basic ops
#   --disable_skip_group_norm:    decompose SkipGroupNorm
#   --disable_embed_layer_norm:   decompose EmbedLayerNorm
#   --disable_bias_splitgelu:     decompose BiasSplitGELU
#   --disable_bias_skip_layer_norm: decompose BiasSkipLayerNorm
#   --disable_bias_gelu:          decompose BiasGELU
#   --disable_packed_kv:          unpack packed key/value tensors
#   --disable_packed_qkv:         unpack packed QKV tensors
#   --no_attention_mask:          simplify attention mask handling
#   --float16:                    convert weights to FP16
#
# The comment above lists all available options; the ones used here are chosen
# to maximize compatibility with ONNX Runtime Web (which lacks custom ops)
# ---------------------------------------------------------------------------
python -m onnxruntime.transformers.models.stable_diffusion.optimize_pipeline -i $model -o $out --overwrite --float16 $opt

# ---------------------------------------------------------------------------
# Step 4: Wrap the optimized models with FP16 I/O casting
#
# Even though --float16 was passed to the optimizer, the graph boundaries
# may still be FP16. Wrap them with Cast nodes so the external interface
# is FP32, which is what ONNX Runtime Web expects.
# ---------------------------------------------------------------------------
python onnx/onnx-wrap-fp16.py --input $out/unet/model.onnx --output  $out/unet/model.onnx
python onnx/onnx-wrap-fp16.py --input $out/vae_decoder/model.onnx --output  $out/vae_decoder/model.onnx
python onnx/onnx-wrap-fp16.py --input $out/text_encoder/model.onnx --output  $out/text_encoder/model.onnx