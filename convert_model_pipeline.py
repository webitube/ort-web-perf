import os
import shutil
import subprocess
from onnxruntime.quantization import quantize_dynamic, QuantType

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def quantize_to_int8(input_path, output_path):
    print(f"Quantizing {input_path} to INT8...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        per_channel=True,
        reduce_range=True,
        weight_type=QuantType.QUInt8
    )

def process_model(model_name, safetensors_path):
    base_fp16 = f"{model_name}-fp16"
    base_int8 = f"{model_name}-int8"
    
    # 1. Export to ONNX (FP16)
    # Using --dtype fp16 as per the new optimum-cli documentation
    # The output directory is a positional argument
    print("--- Exporting to ONNX ---")
    run_cmd([
        "optimum-cli", "export", "onnx", 
        "--model", safetensors_path, 
        "--dtype", "fp16",
        base_fp16
    ])
    
    # 2. Optimize ONNX files
    opt_flags = [
        "--no_attention_mask", "--disable_skip_layer_norm", "--disable_attention", 
        "--disable_nhwc_conv", "--disable_group_norm", "--disable_skip_group_norm", 
        "--disable_embed_layer_norm", "--disable_bias_splitgelu", 
        "--disable_bias_skip_layer_norm", "--disable_bias_gelu", 
        "--disable_packed_kv", "--disable_packed_qkv"
    ]
    
    for comp in ["unet", "vae_decoder", "text_encoder"]:
        comp_path = os.path.join(base_fp16, comp, "model.onnx")
        if os.path.exists(comp_path):
            opt_path = os.path.join(base_fp16, comp, "model_opt.onnx")
            run_cmd(["python", "-m", "onnxruntime.transformers.optimizer", 
                     "--input", comp_path, "--output", opt_path, 
                     "--model_type", "sd_unet" if comp == "unet" else "other", 
                     "--float16"] + opt_flags)
            os.replace(opt_path, comp_path)

    # 3. Create INT8 directory and quantize
    print("--- Creating INT8 folder ---")
    if os.path.exists(base_int8):
        shutil.rmtree(base_int8)
    shutil.copytree(base_fp16, base_int8)
    
    unet_path = os.path.join(base_int8, "unet", "model.onnx")
    if os.path.exists(unet_path):
        quantize_to_int8(unet_path, unet_path)

    print(f"Done! Pipeline ready in {base_fp16} and {base_int8}")