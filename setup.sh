#!/bin/bash

python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -r requirements.txt
deactivate

echo "Libraries installed. Please make sure precomputed spectrograms can be found before proceeding with job submission"
