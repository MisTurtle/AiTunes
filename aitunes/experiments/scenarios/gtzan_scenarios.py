import h5py
import numpy as np
import torch.optim as optim

from os import path

from aitunes.modules import CVAE
from aitunes.audio_processing import PreprocessingCollection

from aitunes.experiments.cases import GtzanExperiment
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario

from aitunes.utils import download_and_extract
from aitunes.utils.loss_functions import simple_mse_kl_loss
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures, precompute_spectrograms_for_audio_folder


class GtzanReconstructionScenarios(AudioBasedScenarioContainer):

    @staticmethod
    def get_path_to_history_root():
        return path.join("history", "gtzan")

    @staticmethod
    def get_path_to_release_root():
        return path.join("assets", "Models", "gtzan")

    @staticmethod
    def get_identifier():
        return "GtzanReconstruction"

    @staticmethod
    def get_description():
        return "Our original training dataset for music generation. It is composed of 1000 compositions accross 10 different music styles."

    def __init__(self):
        super().__init__()
        self.audio_duration = 10.  # Seconds
        self.low_mode = LowResolutionAudioFeatures(self.audio_duration)
        self.high_mode = HighResolutionAudioFeatures(self.audio_duration)
        self.training_file: h5py.File = None
        self.evaluation_file: h5py.File = None
    
    def free_resources(self):
        if self.training_file is not None:
            self.training_file.close()
        if self.evaluation_file is not None:
            self.evaluation_file.close()
    
    def generate_datasets(self):
        if not path.exists(path_to_audios):
            # Download and extract the dataset
            download_and_extract(  # TODO: This is the only thing that differs, merge with sinewave scenarios somehow
                url="https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification",
                target_path=path_to_dataset,
                zip_path=path.join(path_to_dataset, "..", "GTZAN_Dataset.zip"),
                final_size=1241.20,  # Dataset final size is 1.241 GB
                standalone_zipped_dir="Data",  # A Data folder is at the root of the downloaded zip file
                clean=True  # Cleanup the downloaded zip file once finished
            )
        self.free_resources()
        if self.get_mode() == self.high_mode:
            precompute_spectrograms_for_audio_folder(path_to_audios, path_to_training_spectrograms[0], path_to_eval_spectrograms[0], 0.05, self.high_mode, preprocess_audio, preprocess_spectrogram)
            self.training_file, self.evaluation_file = h5py.File(path_to_training_spectrograms[0], mode='r+'), h5py.File(path_to_eval_spectrograms[0], mode='r+')
        elif self.get_mode() == self.low_mode:
            precompute_spectrograms_for_audio_folder(path_to_audios, path_to_training_spectrograms[1], path_to_eval_spectrograms[1], 0.05, self.low_mode, preprocess_audio, preprocess_spectrogram)
            self.training_file, self.evaluation_file = h5py.File(path_to_training_spectrograms[1], mode='r+'), h5py.File(path_to_eval_spectrograms[1], mode='r+')

    def instantiate(self, s, model_path):
        self.set_mode(None)
        model, loss, optimizer = s(self)
        self.generate_datasets()
        return GtzanExperiment(model, model_path or s.model_path, loss, optimizer, self.training_file, self.evaluation_file, self.get_mode())
    
    @scenario(name="GTZAN CVAE", version="1.0-LOW16", description="First real attempt at learning from the GTZAN dataset after model structures have been fixed. Low quality is used for faster training. Latent space size is 16 and only 3 convolutionnal layers are used.")
    def cvae_core16(self):
        self.set_mode(self.low_mode)
        model = CVAE(
            input_shape=[1, *self.get_mode().spectrogram_size],
            conv_filters=[     32,      64,  128],
            conv_kernels=[      3,       3,    3],
            conv_strides=[ (2, 1),  (2, 1),    2],
            latent_space_dim=16
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer

  
    @scenario(name="GTZAN CVAE", version="1.1-LOW16", description="Since attempts with the 1.0-LOW16 are yielding decent results for the network size, this scenario increases the network size from 3 to 6 convolutional layers, keeping the latent space with a size of 16.")
    def cvae_core16_6conv(self):
        self.set_mode(self.low_mode)
        model = CVAE(
            input_shape=[1, *self.get_mode().spectrogram_size],
            conv_filters=[     32,      64,  128, 256, 512, 1024],
            conv_kernels=[      3,       3,    3,   3,   3,    3],
            conv_strides=[ (2, 1),  (2, 1),    2,   2,   2,    2],
            latent_space_dim=16
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="GTZAN CVAE", version="1.1-LOW32", description="Conv layers are increased from 3 to 6, and the latent space to 32.")
    def cvae_core32_6conv(self):
        self.set_mode(self.low_mode)
        model = CVAE(
            input_shape=[1, *self.get_mode().spectrogram_size],
            conv_filters=[     32,      64,  128, 256, 512, 1024],
            conv_kernels=[      3,       3,    3,   3,   3,    3],
            conv_strides=[ (2, 1),  (2, 1),    2,   2,   2,    2],
            latent_space_dim=32
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
        
    @scenario(name="GTZAN CVAE", version="1.2-LOW32", description="")
    def cvae_core32_6conv_2strides(self):
        self.set_mode(self.low_mode)
        model = CVAE(
            input_shape=[1, *self.get_mode().spectrogram_size],
            conv_filters=[ 32, 64, 128, 256, 512, 1024],
            conv_kernels=[  3,  3,   3,   3,   3,    3],
            conv_strides=[  (1, 2),  (1, 2),   (1, 2),   2,   2,    2],
            latent_space_dim=32
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer


    def __del__(self):
        self.free_resources()


def preprocess_audio(y: np.ndarray):
    """
    Preprocessing applied directly to the audio data before precomputing the spectrogram
    :param y: The raw audio data
    """
    return y

def preprocess_spectrogram(y: np.ndarray):
    """
    Preprocessing applied to the spectrogram before being saved as a preprocessed numpy array
    :param y: The spectrogram data
    """
    return PreprocessingCollection.normalise(y, 0, 1)

# Path to the dataset and precomputed mel spectograms (high_res_path, low_res_path)
path_to_dataset = path.join("assets", "Samples", "GTZAN")  # Dataset source path
path_to_audios = path.join(path_to_dataset, "genres_original")  # Path to the dataset's audio files
path_to_spectrograms = path.join(path_to_dataset, "spectrograms")
path_to_training_spectrograms = path.join(path_to_spectrograms, "training_high.h5"), path.join(path_to_spectrograms, "training_low.h5")
path_to_eval_spectrograms = path.join(path_to_spectrograms, "eval_high.h5"), path.join(path_to_spectrograms, "eval_low.h5")
