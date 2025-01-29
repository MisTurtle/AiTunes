import time
import numpy as np
from itertools import cycle

loading_cycle = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
loading_char = next(loading_cycle)
last_call, switch_loading_char_every = 0, 0.05  # seconds

def get_loading_char():
    global last_call, loading_char
    if time.time() - last_call >= switch_loading_char_every:
        last_call, loading_char = time.time(), next(loading_cycle)
    return loading_char

def normalize(nparray):
    return (nparray - np.min(nparray)) / (np.max(nparray) - np.min(nparray))
