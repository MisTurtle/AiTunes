import sys
import os
import torch

# Add `aitunes` to the system path
pkg_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(pkg_dir))

# Make torch use the gpu if available
# if torch.cuda.is_available():
#     torch.set_default_device('cuda')
