import h5py
import numpy as np
import torch.optim as optim

from os import path

from aitunes.modules import CVAE
from aitunes.audio_generation.simple_audio_streams import generate_dataset_of_simple_instruments
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
from aitunes.experiments.cases import SinewaveExperiment
from aitunes.audio_processing import PreprocessingCollection
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures, precompute_spectrograms_for_audio_folder
from aitunes.utils.loss_functions import simple_mse_kl_loss


class SinewaveReconstructionScenarios(AudioBasedScenarioContainer):

    @staticmethod
    def get_path_to_history_root():
        return path.join("history", "sinewave")

    @staticmethod
    def get_path_to_release_root():
        return path.join("assets", "Models", "sinewave")
    
    @staticmethod
    def get_identifier():
        return "SinewaveReconstruction"

    @staticmethod
    def get_description():
        return "Spectrograms of simple sine compound waves are attempted to be reconstructed after being compressed. This experiment aims to find a decent working architecture for audio creation to start tests on the GTZAN dataset which contains actual music."

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
        # Generate Audio Files with high resolution settings
        global path_to_audios, path_to_training_spectrograms, path_to_eval_spectrograms
        if not path.exists(path_to_audios):
            generate_dataset_of_simple_instruments(path_to_audios, sample_rate=self.high_mode.sample_rate, unit_duration=self.audio_duration, unit_per_type=100)
        
        self.free_resources()
        if self.get_mode() == self.high_mode:
            precompute_spectrograms_for_audio_folder(path_to_audios, path_to_training_spectrograms[0], path_to_eval_spectrograms[0], 0.10, self.high_mode, preprocess_audio, preprocess_spectrogram)
            self.training_file, self.evaluation_file = h5py.File(path_to_training_spectrograms[0], mode='r+'), h5py.File(path_to_eval_spectrograms[0], mode='r+')
        elif self.get_mode() == self.low_mode:
            precompute_spectrograms_for_audio_folder(path_to_audios, path_to_training_spectrograms[1], path_to_eval_spectrograms[1], 0.10, self.low_mode, preprocess_audio, preprocess_spectrogram)
            self.training_file, self.evaluation_file = h5py.File(path_to_training_spectrograms[1], mode='r+'), h5py.File(path_to_eval_spectrograms[1], mode='r+')

    def instantiate(self, s, model_path):
        self.set_mode(None)
        model, loss, optimizer = s(self)
        self.generate_datasets()
        return SinewaveExperiment(model, model_path or s.model_path, loss, optimizer, self.training_file, self.evaluation_file, self.get_mode(), flatten=not isinstance(model, CVAE))
    
    @scenario(name="Convolutional VAE", version="1.0-LOW4", description="Scenarios in this series aim to find a decent latent space size. 6 convolutional layers with asymmetrical strides at the beginning are used.\nLatent Space Size : 4 dimensions")
    def cvae_core4(self):
        self.set_mode(self.low_mode)
        model = CVAE(
            input_shape=[1, *self.get_mode().spectrogram_size],
            conv_filters=[     32,      64,  128, 256, 512, 1024],
            conv_kernels=[      3,       3,    3,   3,   3,    3],
            conv_strides=[ (2, 1),  (2, 1),    2,   2,   2,    2],
            latent_space_dim=4
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer

    @scenario(name="Convolutional VAE", version="1.0-LOW8", description="Scenarios in this series aim to find a decent latent space size. 6 convolutional layers with asymmetrical strides at the beginning are used.\nLatent Space Size : 8 dimensions\nBeta = 1")
    def cvae_core8(self):
        self.set_mode(self.low_mode)
        model = CVAE(
            input_shape=[1, *self.get_mode().spectrogram_size],
            conv_filters=[     32,      64,  128, 256, 512, 1024],
            conv_kernels=[      3,       3,    3,   3,   3,    3],
            conv_strides=[ (2, 1),  (2, 1),    2,   2,   2,    2],
            latent_space_dim=8
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Convolutional VAE", version="1.0-LOW16", description="Scenarios in this series aim to find a decent latent space size. 6 convolutional layers with asymmetrical strides at the beginning are used.\nLatent Space Size : 16 dimensions\nBeta = 1")
    def cvae_core16(self):
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
path_to_dataset = path.join("assets", "Samples", "SINEWAVE")  # Dataset source path
path_to_audios = path.join(path_to_dataset, "audio")  # Path to the dataset's audio files
path_to_spectrograms = path.join(path_to_dataset, "spectrograms")
path_to_training_spectrograms = path.join(path_to_spectrograms, "training_high.h5"), path.join(path_to_spectrograms, "training_low.h5")
path_to_eval_spectrograms = path.join(path_to_spectrograms, "eval_high.h5"), path.join(path_to_spectrograms, "eval_low.h5")
