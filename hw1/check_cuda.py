import torch

print(
    torch.cuda.is_available(), 
    torch.cuda.device_count(), 
    torch.cuda.current_device()
    )