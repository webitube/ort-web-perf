import torch
import onnxruntime as ort

print(torch.cuda.is_available())
print(torch.version.cuda)

print('ORT version:', ort.__version__)
print('Providers:', ort.get_available_providers())