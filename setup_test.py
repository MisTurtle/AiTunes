import torch
import aitunes

print('> # Torch Devices:', torch.cuda.device_count() + torch.cpu.device_count())
print('> Cuda:', 'Enabled' if torch.cuda.is_available() else 'Disabled')
print('> # Cuda Devices:', torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print('> Selected Cuda Device:', torch.cuda.current_device(), '(%s)' % torch.cuda.get_device_name())

test_tensor = torch.zeros((100, 5, 5))
print('> Default device:', test_tensor.device.type)

