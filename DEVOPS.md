# DEVOPS - Testing the CLI Tool

## Quick Start

```powershell
conda activate ort-web-perf
python test-model-cli.py --model-path sd-turbo-ort-web --prompt "a photo of an astronaut riding a horse" --output test-sd-turbo.png --seed 42
```

## Prerequisites

- **Conda environment**: `ort-web-perf` with `onnxruntime`, `transformers`, `numpy`
- **ONNX model directory** with required structure:
  ```
  model-path/
  ├── unet/model.onnx
  ├── text_encoder/model.onnx
  ├── vae_decoder/model.onnx
  └── tokenizer/
      ├── vocab.json
      └── merges.txt
  ```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-path` | Yes | - | Path to ONNX model directory |
| `--prompt` | No | "a photo of an astronaut riding a horse" | Text prompt |
| `--output` | No | "test-output.png" | Output PNG path |
| `--seed` | No | 42 | Random seed |
| `--width` | No | 512 | Image width (multiple of 64) |
| `--height` | No | 512 | Image height (multiple of 64) |
| `--provider` | No | "auto" | `auto` / `cpu` / `cuda` / `tensorrt` |

## Provider Selection

| Provider | Use Case | Notes |
|----------|----------|-------|
| `auto` | Default | TensorRT → CUDA → CPU fallback chain |
| `cuda` | GPU only | CUDAExecutionProvider only |
| `cpu` | CPU only | Fails on FP16 models |
| `tensorrt` | GPU + TRT | Requires TensorRT libs installed |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `tensor(float16)` type error | FP16 model on CPU | Use `--provider cuda` |
| `nvinfer_10.dll missing` | TensorRT not installed | Use `--provider cuda` instead |
| `Missing files` | Wrong model path | Check directory structure |
| `Execution provider not available` | GPU not found | Use `--provider cpu` with FP32 model |

## Test Matrix

```powershell
# Basic smoke test
python test-model-cli.py --model-path sd-turbo-ort-web --output test.png

# GPU test
python test-model-cli.py --model-path sd-turbo-ort-web --provider cuda --output test-gpu.png

# Custom prompt
python test-model-cli.py --model-path sd-turbo-ort-web --prompt "a castle in the clouds" --output castle.png

# Different seed
python test-model-cli.py --model-path sd-turbo-ort-web --seed 123 --output test-seed123.png
```

## CI/CD Integration

```yaml
# Example GitHub Actions step
- name: Test ONNX model
  run: |
    conda activate ort-web-perf
    python test-model-cli.py `
      --model-path sd-turbo-ort-web `
      --prompt "a photo of an astronaut riding a horse" `
      --output test-output.png `
      --seed 42
  shell: pwsh
```

## Performance Baseline (RTX 4090, CUDA)

| Stage | Time |
|-------|------|
| text_encoder load | ~3.7s |
| unet load | ~8.8s |
| vae_decoder load | ~0.3s |
| text_encoder inference | ~0.3s |
| unet inference | ~0.5s |
| vae_decoder inference | ~0.3s |
| **Total** | **~0.4s** (warm) |
