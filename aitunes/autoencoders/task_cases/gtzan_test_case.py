from aitunes.autoencoders.task_cases import AutoencoderTaskCase, FLAG_NONE

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

class GtzanDatasetTestCase(AutoencoderTaskCase):
    """
    The GTZAN dataset has 100 30-second samples for a wapping 10 different genres, resulting in 1000 * 30 = 30'000 seconds of music to train on
    For this purpose, 95 samples will be split and used for the training process in each genre, while the other 5 will be used for model evaluation
    """
    pass
