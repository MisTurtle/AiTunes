from aitunes.utils import download_and_extract, get_loading_char, read_labelled_folder
from aitunes.autoencoders.task_cases import AutoencoderTaskCase, FLAG_NONE
from aitunes.audio_processing import PreprocessingCollection, AudioProcessingInterface

from os import path, makedirs, listdir, rmdir
from shutil import move

import zipfile
import requests
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

class GtzanDatasetTestCase(AutoencoderTaskCase):
    """
    The GTZAN dataset has 100 30-second samples for a wapping 10 different genres, resulting in 1000 * 30 = 30'000 seconds of music to train on
    For this purpose, 95 samples will be split and used for the training process in each genre, while the other 5 will be used for model evaluation
    """
    def __init__(self, model, weights_path, loss, optimizer, preprocessing_fn, flatten: bool = False, flags: int = FLAG_NONE):
        """
        :param preprocessing_fn: A function taking a path to a .wav file and returns ready-to-go audio data.
        """
        super().__init__("GTZAN", model, weights_path, loss, optimizer, flags)
        
        self._dataset_path = path.join("assets", "Samples", "GTZAN")
        self._flatten = flatten
        self._preprocessing_fn = preprocessing_fn
        self.train_loader = []
        self.test_loader = []

        self._download_dataset()

        for label, files in read_labelled_folder(path.join(self._dataset_path, "genres_original"), ".wav").items():
            train_batch = files[:-5] if len(files) > 5 else files
            test_batch = files[-5:] if len(files) > 5 else files
            self.train_loader += [(label, file) for file in train_batch]
            self.test_loader += [(label, file) for file in test_batch]
        
        self.add_middleware(self.save_verification_extract)
    
    def _download_dataset(self):
        download_and_extract(
            url="https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification",
            target_path=self._dataset_path,
            zip_path=path.join(self._dataset_path, "..", "GTZAN_Dataset.zip"),
            final_size=1241.20,
            standalone_zipped_dir="Data",
            clean=True
        )

    def save_verification_extract(self, og, pred, embeds, labels, args):
        for i, label in enumerate(labels[0]):
            og_item = np.reshape(og[i].cpu().numpy(), (32, -1))
            pred_item = np.reshape(pred[i].cpu().numpy(), (32, -1))
            
            p_i = AudioProcessingInterface.create_for(f"assets/Samples/generated/gtzan/generated_{i}.wav", mode="log_mel", sr=22050, data=pred_item)
            p_i.save(None).summary(qualifier=f"Prediction {i}")
            
            og_i = AudioProcessingInterface.create_for(f"assets/Samples/generated/gtzan/original_{i}.wav", mode="log_mel", sr=22050, data=og_item)
            og_i.save(None).summary(qualifier=f"Original {i}")

        a = input()
        
    def next_batch(self, training): 
        dataset = self.train_loader
        if not training:
            dataset = self.test_loader
        
        random.shuffle(dataset)
        batch_size = 5
        for i in range(0, len(dataset) // batch_size):
            items = dataset[i * batch_size:(i + 1) * batch_size]
            labels, file_paths = zip(*items)
            preprocessed_audio_files = np.array(list(filter(lambda x: x is not None, map(self._preprocessing_fn, file_paths))))
            audio_data = torch.tensor(preprocessed_audio_files, dtype=torch.float32)

            if self._flatten:
                audio_data = audio_data.flatten(start_dim=1, end_dim=2)
            else:
                audio_data = audio_data.unsqueeze(1)
            yield audio_data, labels

    def interactive_evaluation(self):
        raise NotImplementedError

        
        
