from typing import Union
from itertools import cycle
from os import path, makedirs, listdir, rmdir, remove
from shutil import move

import os
import time
import string
import zipfile
import h5py
import numpy as np
import random
import requests
import torch
import torch.nn.functional as F

quiet: bool = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def download_and_extract(url: str, target_path: str, zip_path: Union[str, None] = None, final_size: Union[float, None] = None, standalone_zipped_dir: Union[str, None] = None, clean: bool = True):
    """
    Download and extract a zip file to a given path
    :param url: Remote url to download from
    :param target_path: Where to download the contents
    :param zip_path: Temporary zip path to write to before extracting to `target_path`. A random string is generated if None is given
    :param final_size: Expected download size (in MB), only useful for visual purpose
    :param standalone_zipped_dir: Name of the dir if the root of the zip file only contains one folder, take the contents of that folder and move it upwards one rank
    :param clean: Temporary zip file gets deleted if True
    """
    if path.exists(target_path) and len(listdir(target_path)) > 0:
        print(f"Skipped download for {url}: Items exist at path {target_path}")
        return
    
    if zip_path is None:
        zip_path = path.join(target_path, "..", random.choices(string.ascii_letters, 9) + ".zip")
        makedirs(path.dirname(zip_path), exist_ok=True)
    
    print(f"Downloading from {url} to {zip_path}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download from {url} (Error {response.status_code})")
    
    one_mb = 1024 * 1024
    dl_size = 0
    dl_suffix = lambda x: "" if final_size is None else f" / {final_size:.2f} MB ({100 * x / final_size:.2f}%)"
    try:
        with open(path.abspath(zip_path), "wb") as file:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    dl_size += len(chunk) / one_mb
                    file.write(chunk)
                    if not quiet:
                        print(f"\r{get_loading_char()} {dl_size:.2f} MB{dl_suffix(dl_size)}", end='')

        print(f"\rDownload complete. Extracting to {target_path}")
        makedirs(target_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(target_path)
        
        if standalone_zipped_dir is not None:
            data_dir_to_rm = path.join(target_path, standalone_zipped_dir)
            for filename in listdir(data_dir_to_rm):
                move(path.join(data_dir_to_rm, filename), path.join(target_path))
            rmdir(data_dir_to_rm)

        print(f"Successfully extracted to {target_path}")
    finally:
        if clean and path.exists(zip_path):
            remove(zip_path)

def save_dataset(path_to: str, datasets: dict, attrs: dict = {}):
    makedirs(path.dirname(path_to), exist_ok=True)
    with h5py.File(path_to, "w") as f:
        for name, values in datasets.items():
            f.create_dataset(name, data=values)
        for key, val in attrs.items():
            f.attrs[key] = val

def append_to_dataset(path_to: str, datasets: dict):
    makedirs(path.dirname(path_to), exist_ok=True)
    with h5py.File(path_to, 'a') as f:
        for key, value in datasets.items():
            if key in f:
                f[key].resize((f[key].shape[0] + value.shape[0]), axis=0)
                f[key][-value.shape[0]:] = value
            else:
                maxshape = (None,) + value.shape[1:]
                f.create_dataset(key, data=value, maxshape=maxshape, chunks=True)

