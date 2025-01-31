from aitunes.utils import download_and_extract, read_labelled_folder
from aitunes.autoencoders.task_cases import AutoencoderTaskCase, FLAG_NONE
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
    def __init__(self, model, weights_path, loss, optimizer, training_data: h5py.File, evaluation_data: h5py.File, flatten: bool = False, flags: int = FLAG_NONE):
        """
        :param preprocessing_fn: A function taking a path to a .wav file and returns ready-to-go audio data.
        """
        super().__init__("GTZAN", model, weights_path, loss, optimizer, flags)
        self._flatten = flatten
        self.train_loader = training_data
        self.test_loader = evaluation_data
        random.shuffle(self.train_loader)
        random.shuffle(self.test_loader)
        
        self.add_middleware(self.save_verification_extract)
    

    def save_verification_extract(self, og, pred, embeds, labels, args):
        # for i, label in enumerate(labels[0]):
        #     og_item = np.reshape(og[i].cpu().numpy(), (128, -1))
        #     pred_item = np.reshape(pred[i].cpu().numpy(), (128, -1))
            
        #     p_i = AudioProcessingInterface.create_for(f"assets/Samples/generated/gtzan/generated_{i}.wav", mode="log_mel", sr=22050, data=pred_item)
        #     p_i.save(None).summary(qualifier=f"Prediction {i}")
            
        #     og_i = AudioProcessingInterface.create_for(f"assets/Samples/generated/gtzan/original_{i}.wav", mode="log_mel", sr=22050, data=og_item)
        #     og_i.save(None).summary(qualifier=f"Original {i}")

        # a = input()
        pass  # TODO

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
            current_index += batch_size
            spectrograms = torch.tensor(spectrograms, dtype=torch.float32)

            if self._flatten:
                spectrograms = spectrograms.flatten(start_dim=1, end_dim=2)
            else:
                spectrograms = spectrograms.unsqueeze(1)

            yield spectrograms, np.zeros((spectrograms.shape[0], ))  # No labels passed, might change later

    def interactive_evaluation(self):
        raise NotImplementedError

        
        
