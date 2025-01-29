from aitunes.utils import read_labelled_folder
from aitunes.autoencoders.task_cases import AutoencoderTaskCase, FLAG_NONE

import time
import random
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

import torch

class GtzanDatasetTestCase(AutoencoderTaskCase):
    """
    The GTZAN dataset has 100 30-second samples for a wapping 10 different genres, resulting in 1000 * 30 = 30'000 seconds of music to train on
    For this purpose, 95 samples will be split and used for the training process in each genre, while the other 5 will be used for model evaluation
    """
    def __init__(self, model, weights_path, loss, optimizer, preprocessing_fn, flags: int = FLAG_NONE):
        """
        :param preprocessing_fn: A function taking a path to a .wav file and returns ready-to-go audio data.
        """
        super().__init__("GTZAN", model, weights_path, loss, optimizer, flags)
        
        self._preprocessing_fn = preprocessing_fn
        self.train_loader = []
        self.test_loader = []

        for label, files in read_labelled_folder(path.join("assets", "Samples", "GTZAN", "genres_original"), ".wav").items():
            train_batch = files[:-5] if len(files) > 5 else files
            test_batch = files[-5:] if len(files) > 5 else files
            self.train_loader += [(label, file) for file in train_batch]
            self.test_loader += [(label, file) for file in test_batch]
        
    def next_batch(self, training): 
        dataset = self.train_loader
        if not training:
            dataset = self.test_loader
        
        random.shuffle(dataset)
        batch_size = 10
        for i in range(0, len(dataset) // batch_size):
            items = dataset[i * batch_size:(i + 1) * batch_size]
            labels, file_paths = zip(*items)
            audio_data = torch.tensor(list(map(self._preprocessing_fn, file_paths)), dtype=torch.float32)
            yield audio_data, labels

    def interactive_evaluation(self):
        raise NotImplementedError

        
        
