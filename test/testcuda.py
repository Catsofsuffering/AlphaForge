import torch

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Available devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')
