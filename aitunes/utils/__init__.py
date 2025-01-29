import time
import numpy as np

import os
import os.path as path
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

def read_labelled_folder(path_to_root: str, ext: str = "") -> dict[list[str]]:
    """
    Returns a dict whose key is the label and items are samples for said label
    :param path_to_root: Path to the root folder
    :param ext: Filter files by extension, only returning those that match that ext
    """
    if not path.exists(path_to_root):
        raise FileNotFoundError(f"Reading labelled folder at {path_to_root} failed: Folder does not exist")
    if not path.isdir(path_to_root):
        raise EOFError(f"Path {path_to_root} points to a file: Directory expected")

    ordered_per_label = {}
    for label in os.listdir(path_to_root):
        label_path = path.join(path_to_root, label)
        if not path.isdir(label_path):
            continue
        
        ordered_per_label[label] = []
        for labelled_item in os.listdir(label_path):
            if not labelled_item.endswith(ext):
                continue

            ordered_per_label[label].append(path.join(label_path, labelled_item))
    
    return ordered_per_label
