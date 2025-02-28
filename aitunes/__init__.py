import torch

# Make torch use the gpu if available
if torch.cuda.is_available():
    torch.set_default_device('cuda')
torch.manual_seed(0)

import sys
import signal
def handle_signal(signum, frame):
    print("Imminent termination signal received, exiting early to save progress...")
    quit(0)

if sys.platform == "linux":
    signal.signal(signal.SIGUSR1, handle_signal)
else:
    signal.signal(signal.SIGINT, handle_signal)

from .audio_generation import *
from .audio_processing import *
from .modules import *
from .utils import *
