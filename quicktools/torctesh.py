import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(hasattr(torch.amp, "grad_scaler"))

