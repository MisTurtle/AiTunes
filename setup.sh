#!/bin/bash

# Install required libraries for 
# Make sure to remove torch, torchvision and torchaudio from requirements.txt before running it
# Otherwise the versions supporting cuda will be overwritten

python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
deactivate
