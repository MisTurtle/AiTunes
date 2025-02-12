import h5py
import numpy as np

from os import path

from aitunes.experiments.autoencoder_experiment import SpectrogramBasedAutoencoderExperiment
from aitunes.audio_processing import PreprocessingCollection
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer

from aitunes.utils import download_and_extract
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures, precompute_spectrograms_for_audio_folder


class FmaReconstructionScenarios(AudioBasedScenarioContainer):

    @staticmethod
    def get_path_to_history_root():
        return path.join("history", "fma")

    @staticmethod
    def get_path_to_release_root():
        return path.join("assets", "Models", "fma")

    @staticmethod
    def get_identifier():
        return "FmaReconstruction"

    @staticmethod
    def get_description():
        return "A significantly larger dataset for training on music of various types. Due to its size, only production-grade models will be trained on this dataset."

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
            download_and_extract(
                url="https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
                target_path=path_to_dataset,
                zip_path=path.join(path_to_dataset, "..", "FMA.zip"),
                final_size=22721.29,  # Dataset final size is 22'721 MB
                standalone_zipped_dir=None,  # A Data folder is at the root of the downloaded zip file
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
        return SpectrogramBasedAutoencoderExperiment("FMA", model, model_path or s.model_path, loss, optimizer, self.training_file, self.evaluation_file, self.get_mode(), 16)

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
path_to_dataset = path.join("assets", "Samples", "FMA")  # Dataset source path
path_to_audios = path.join(path_to_dataset, "fma_medium")  # Path to the dataset's audio files
path_to_spectrograms = path.join(path_to_dataset, "spectrograms")
path_to_training_spectrograms = path.join(path_to_spectrograms, "training_high.h5"), path.join(path_to_spectrograms, "training_low.h5")
path_to_eval_spectrograms = path.join(path_to_spectrograms, "eval_high.h5"), path.join(path_to_spectrograms, "eval_low.h5")
