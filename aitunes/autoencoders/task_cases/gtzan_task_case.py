from aitunes.utils import download_and_extract, read_labelled_folder
from aitunes.autoencoders.task_cases import AutoencoderTaskCase, FLAG_NONE, Middleware
from aitunes.audio_processing import AudioProcessingInterface

from os import path
import h5py
import random
import numpy as np
import torch

class GtzanDatasetTaskCase(AutoencoderTaskCase):
    """
    The GTZAN dataset has 100 30-second samples for a wapping 10 different genres, resulting in 1000 * 30 = 30'000 seconds of music to train on
    For this purpose, 95 samples will be split and used for the training process in each genre, while the other 5 will be used for model evaluation
    """
    def __init__(self, model, weights_path, loss, optimizer, training_data: h5py.File, evaluation_data: h5py.File, compare_audio: Middleware, flatten: bool = False, flags: int = FLAG_NONE):
        """
        :param compare_audio: A middleware taking in the original normalized spectrogram and the one reconstructed by the model, and generating audio out of those
        """
        super().__init__("GTZAN", model, weights_path, loss, optimizer, flags)
        self._flatten = flatten
        self.train_loader = training_data
        self.test_loader = evaluation_data
        
        self.add_middleware(compare_audio)

    def next_batch(self, training): 
        dataset = self.train_loader
        if not training:
            dataset = self.test_loader
        
        complete = False
        batch_size, current_index = 50, 0
        while not complete:
            if current_index + batch_size >= dataset.shape[0]:
                spectrograms = dataset[current_index:]
                complete = True
            else:
                spectrograms = dataset[current_index:current_index + batch_size]
            spectrograms = torch.tensor(spectrograms, dtype=torch.float32)

            if self._flatten:
                spectrograms = spectrograms.flatten(start_dim=1, end_dim=2)
            else:
                spectrograms = spectrograms.unsqueeze(1)

            yield spectrograms, np.arange(current_index, current_index + spectrograms.shape[0])  # No real labels passed, but ids to the spectrograms.
            current_index += batch_size

    def interactive_evaluation(self):
        raise NotImplementedError

        
        
