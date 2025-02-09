import torch

# Make torch use the gpu if available
if torch.cuda.is_available():
    torch.set_default_device('cuda')

from .audio_generation import *
from .audio_processing import *
from .modules import *
from .utils import *
