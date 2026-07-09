# **Automated Model Pipeline: Safetensors to Web-Compatible ONNX**

### Optimum ONNX: How-To Guides

* Export a model to ONNX: `https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model`
* Add support for exporting an architecture to ONNX: `https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/contribute`

#### optimum-cli options

```
optimum-cli export onnx --help

usage: optimum-cli export onnx [-h] -m MODEL [--task TASK] [--opset OPSET] [--device DEVICE] [--dtype {fp32,fp16,bf16}] [--optimize {O1,O2,O3,O4}] [--monolith]
                               [--no-post-process] [--variant VARIANT] [--framework {pt}] [--atol ATOL] [--cache_dir CACHE_DIR] [--trust-remote-code]
                               [--pad_token_id PAD_TOKEN_ID] [--library-name {transformers,diffusers,timm,sentence_transformers}] [--model-kwargs MODEL_KWARGS]
                               [--no-dynamic-axes] [--no-constant-folding] [--slim] [--dynamo] [--batch_size BATCH_SIZE] [--sequence_length SEQUENCE_LENGTH]
                               [--num_choices NUM_CHOICES] [--width WIDTH] [--height HEIGHT] [--num_channels NUM_CHANNELS] [--feature_size FEATURE_SIZE]
                               [--nb_max_frames NB_MAX_FRAMES] [--audio_sequence_length AUDIO_SEQUENCE_LENGTH] [--point_batch_size POINT_BATCH_SIZE]
                               [--nb_points_per_image NB_POINTS_PER_IMAGE] [--visual_seq_length VISUAL_SEQ_LENGTH]
                               output

options:
  -h, --help            show this help message and exit

Required arguments:
  -m MODEL, --model MODEL
                        Model ID on huggingface.co or path on disk to load model from.
  output                Path indicating the directory where to store the generated ONNX model.

Optional arguments:
  --task TASK           The task to export the model for. If not specified, the task will be auto-inferred from the model's metadata or files. For tasks that
                        generate text, add the `xxx-with-past` suffix to export the model using past key values caching. Available tasks depend on the model, but
                        are among the following list: ['audio-classification', 'audio-frame-classification', 'audio-xvector', 'automatic-speech-recognition',
                        'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-segmentation', 'image-
                        text-to-text', 'image-to-image', 'image-to-text', 'inpainting', 'keypoint-detection', 'mask-generation', 'masked-im', 'multiple-choice',
                        'object-detection', 'question-answering', 'reinforcement-learning', 'semantic-segmentation', 'sentence-similarity', 'text-classification',
                        'text-generation', 'text-to-audio', 'text-to-image', 'text2text-generation', 'time-series-forecasting', 'token-classification', 'visual-
                        question-answering', 'zero-shot-image-classification', 'zero-shot-object-detection'].
  --opset OPSET         If specified, ONNX opset version to export the model with. Otherwise, the default opset for the given model architecture will be used.
  --device DEVICE       The device to use to do the export. Defaults to "cpu".
  --dtype {fp32,fp16,bf16}
                        The floating point precision to use for the export. Supported options: fp32 (float32), fp16 (float16), bf16 (bfloat16).
  --optimize {O1,O2,O3,O4}
                        Allows to run ONNX Runtime optimizations directly during the export. Some of these optimizations are specific to ONNX Runtime, and the
                        resulting ONNX will not be usable with other runtime as OpenVINO or TensorRT. Possible options: - O1: Basic general optimizations - O2:
                        Basic and extended general optimizations, transformers-specific fusions - O3: Same as O2 with GELU approximation - O4: Same as O3 with mixed
                        precision (fp16, GPU-only, requires `--device cuda`)
  --monolith            Forces to export the model as a single ONNX file. By default, the ONNX exporter may break the model in several ONNX files, for example for
                        encoder-decoder models where the encoder should be run only once while the decoder is looped over.
  --no-post-process     Allows to disable any post-processing done by default on the exported ONNX models. For example, the merging of decoder and decoder-with-past
                        models into a single ONNX model file to reduce memory usage.
  --variant VARIANT     Select a variant of the model to export.
  --framework {pt}      The framework to use for the export. Defaults to 'pt' for PyTorch.
  --atol ATOL           If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.
  --cache_dir CACHE_DIR
                        Path indicating where to store cache.
  --trust-remote-code   Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in
                        which you have read the code, as it will execute on your local machine arbitrary code present in the model repository.
  --pad_token_id PAD_TOKEN_ID
                        This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it.
  --library-name {transformers,diffusers,timm,sentence_transformers}
                        The library on the model. If not provided, will attempt to infer the local checkpoint's library
  --model-kwargs MODEL_KWARGS
                        Any kwargs passed to the model forward, or used to customize the export for a given model.
  --no-dynamic-axes     Disable dynamic axes during ONNX export
  --no-constant-folding
                        PyTorch-only argument. Disables PyTorch ONNX export constant folding.
  --slim                Enables onnxslim optimization.
  --dynamo              Selects dynamo exporter instead of torch script exporter (Option `dynamo=True` with `torch.onnx.export`),
                        this is the recommended option for opset >= 18.

Input shapes (if necessary, this allows to override the shapes of the input given to the ONNX exporter, that requires an example input).:
  --batch_size BATCH_SIZE
                        Text tasks only. Batch size to use in the example input given to the ONNX export.
  --sequence_length SEQUENCE_LENGTH
                        Text tasks only. Sequence length to use in the example input given to the ONNX export.
  --num_choices NUM_CHOICES
                        Text tasks only. Num choices to use in the example input given to the ONNX export.
  --width WIDTH         Image tasks only. Width to use in the example input given to the ONNX export.
  --height HEIGHT       Image tasks only. Height to use in the example input given to the ONNX export.
  --num_channels NUM_CHANNELS
                        Image tasks only. Number of channels to use in the example input given to the ONNX export.
  --feature_size FEATURE_SIZE
                        Audio tasks only. Feature size to use in the example input given to the ONNX export.
  --nb_max_frames NB_MAX_FRAMES
                        Audio tasks only. Maximum number of frames to use in the example input given to the ONNX export.
  --audio_sequence_length AUDIO_SEQUENCE_LENGTH
                        Audio tasks only. Audio sequence length to use in the example input given to the ONNX export.
  --point_batch_size POINT_BATCH_SIZE
                        For Segment Anything. It corresponds to how many segmentation masks we want the model to predict per input point.
  --nb_points_per_image NB_POINTS_PER_IMAGE
                        For Segment Anything. It corresponds to the number of points per segmentation masks.
  --visual_seq_length VISUAL_SEQ_LENGTH
                        Visual sequence length
```


This project provides an automated pipeline script (`pipeline.py`) designed to convert a single-file safetensors model (containing `UNet`, `CLIP/Text Encoder`, and `VAE`) into a directory structure compatible with `web-txt2img` (based on `onnxruntime-web`).

The pipeline automates the complex steps of `ONNX` conversion, transformer-specific graph optimization, and quantization to `fp16` and `int8`.

## **Architecture**

The pipeline operates in three distinct phases:

1. **Export Phase:** Uses `optimum-cli` to unpack the safetensors model into standard `ONNX` component folders (`unet`, `vae_decoder`, `text_encoder`).  
2. **Optimization Phase:** Applies onnxruntime.transformers.optimizer using custom flags to remove complex layers (e.g., attention masking, custom group norms) that are incompatible with browser-based inference.  
3. **Quantization Phase:** Creates a secondary directory for int8 quantization using onnxruntime.quantization to drastically reduce model size and improve web-loading performance.

## **Prerequisites**

Ensure your environment has the following installed:

* Python 3.8+  
* optimum, transformers, diffusers  
* onnxruntime (or onnxruntime-gpu) (with transformers and quantization modules)

Install dependencies via:



```bash
conda create -n ort-web-perf
conda activate ort-web-perf
conda install -c nvidia cuda-toolkit
pip install numpy==1.26.4 //1.23.5
pip install -U transformers diffusers optimum onnxruntime-gpu transformers onnx "optimum-onnx[onnxruntime-gpu]" accelerate onnxscript
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Note: Ensure that the version of cuda that you install for torch torchvision torchaudio must matctch the first CUDA in the path which, at this time, is 12.4


## **Running the Pipeline**

### **1. Setup**

Place your safetensors model file in a known directory.

### **2. Execution**

Import the `process_model()` function from `pipeline.py` or execute the script directly.

```python
from pipeline import process_model

# Usage: process_model(model_name, path_to_safetensors)  
process_model("sd-turbo", "./models/sd-turbo.safetensors")
```

### **3. Output Structure**

The script generates two output directories in your working folder:

* `[modelname]-fp16`: Contains the optimized, float16 models.  
* `[modelname]-int8`: Contains the optimized, int8-quantized UNet model, ready for high-performance web deployment.

## **Key Optimization Flags**

The script applies the following flags to ensure compatibility with onnxruntime-web:

```
--no_attention_mask, --disable_attention, --disable_nhwc_conv, --disable_group_norm, --disable_packed_kv, etc. 
```

These flags effectively flatten the computational graph to ensure it runs efficiently within the constrained environment of a web browser.