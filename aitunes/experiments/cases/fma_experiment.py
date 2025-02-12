from aitunes.modules.autoencoder_modules import CVAE
from aitunes.utils.audio_utils import AudioFeatures, audio_model_interactive_evaluation
from aitunes.experiments import AutoencoderExperiment

import h5py
import numpy as np
import torch

class FmaExperiment(AutoencoderExperiment):

    """
    The FMA dataset has 25000 30-second samples of 30 seconds, resulting in 208 hours of music to train on
    """
    # TODO : Compare this class with Sinewave experimetn and merge shared properties to reduce boilerplate 
    def __init__(self, model, weights_path, loss, optimizer, training_data: h5py.File, evaluation_data: h5py.File, mode: AudioFeatures):
        super().__init__("FMA", model, weights_path, loss, optimizer)

        self.train_loader, self.train_labels = training_data["spectrograms"], training_data["labels"]
        self.training_indices = np.arange(len(self.train_loader))

        self.test_loader, self.test_labels = evaluation_data["spectrograms"], evaluation_data["labels"]
        self.test_indices = np.arange(len(self.test_loader))

        self.mode = mode

    def next_batch(self, training): 
        if training:
            dataset = self.train_loader
            indices = self.training_indices
        else:
            dataset = self.test_loader
            indices = self.test_indices

        np.random.shuffle(indices)
        
        complete = False
        batch_size, current_index = 16, 0
        while not complete:
            if current_index + batch_size >= dataset.shape[0]:
                batch_indices = np.concatenate((indices[current_index:], indices[:batch_size + current_index - dataset.shape[0]]))
                complete = True
            else:
                batch_indices = indices[current_index:current_index + batch_size]

            batch_indices = np.sort(batch_indices)
            spectrograms = dataset[batch_indices]
            spectrograms = torch.tensor(spectrograms, dtype=torch.float32)

            if self.flatten:
                spectrograms = spectrograms.flatten(start_dim=1, end_dim=2)
            else:
                spectrograms = spectrograms.unsqueeze(1)

            yield spectrograms, batch_indices  # No real labels passed, but ids to the spectrograms.
            current_index += batch_size

    def interactive_evaluation(self):
        self.model.eval()
        with torch.no_grad():
            print("Wait a moment while the first audio is being processed...")
            audio_model_interactive_evaluation(
                features=self.mode,
                test_loader=self.test_loader,
                test_labels=self.test_labels,
                model=self.model,
                loss_criterion=self._loss_criterion
            )