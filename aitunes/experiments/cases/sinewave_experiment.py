from aitunes.utils.audio_utils import AudioFeatures, audio_model_interactive_evaluation
from aitunes.experiments import AutoencoderExperiment

import h5py
import numpy as np
import torch

class SinewaveExperiment(AutoencoderExperiment):

    def __init__(self, model, weights_path, loss, optimizer, training_data: h5py.File, evaluation_data: h5py.File, mode: AudioFeatures, flatten: bool = False):
        super().__init__("SineWave", model, weights_path, loss, optimizer)
        self._flatten = flatten
        
        self.train_loader = training_data["spectrograms"]
        self.train_labels = training_data["labels"]
        self.training_indices = np.arange(len(self.train_loader))

        self.test_loader = evaluation_data["spectrograms"]
        self.test_labels = evaluation_data["labels"]
        self.test_indices = np.arange(len(self.test_loader))

        self.mode = mode

    def next_batch(self, training): 
        if training:
            dataset = self.train_loader
            np.random.shuffle(self.training_indices)
            indices = self.training_indices
        else:
            dataset = self.test_loader
            np.random.shuffle(self.test_indices)
            indices = self.test_indices
        
        complete = False
        batch_size, current_index = 16, 0
        while not complete:
            if current_index + batch_size >= dataset.shape[0]:
                batch_indices = indices[current_index:]
                complete = True
            else:
                batch_indices = indices[current_index:current_index + batch_size]
            
            batch_indices = np.sort(batch_indices)  # Indices need to be sorted for h5py slices
            spectrograms = dataset[batch_indices]
            spectrograms = torch.tensor(spectrograms, dtype=torch.float32)

            if self._flatten:
                spectrograms = spectrograms.flatten(start_dim=1, end_dim=2)
            else:
                spectrograms = spectrograms.unsqueeze(1)

            yield spectrograms, batch_indices
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
