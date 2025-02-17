from os import path
import torch.optim as optim

from aitunes.modules import CVAE
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
from aitunes.utils.loss_functions import simple_mse_kl_loss
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures


class GtzanReconstructionScenarios(AudioBasedScenarioContainer):

    @property
    def path_to_history_root(self):
        return path.join("history", "gtzan")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "gtzan")

    @property
    def identifier(self):
        return "GTZAN"

    @property
    def description(self):
        return "Our original training dataset for music generation. It is composed of 1000 compositions accross 10 different music styles."

    @property
    def path_to_dataset_root(self):
        return path.join("assets", "Samples", "GTZAN")

    @property
    def all_paths_to_training_spectrograms(self):
        return path.join(self.path_to_dataset_root, "spectrograms", "training_high.h5"), path.join(self.path_to_dataset_root, "spectrograms", "training_low.h5")

    @property
    def all_paths_to_eval_spectrograms(self):
        return path.join(self.path_to_dataset_root, "spectrograms", "eval_high.h5"), path.join(self.path_to_dataset_root, "spectrograms", "eval_low.h5")

    @property
    def path_to_audio_root(self):
        return path.join(self.path_to_dataset_root, "genres_original")

    @property
    def all_modes(self):
        return HighResolutionAudioFeatures(5.0), LowResolutionAudioFeatures(5.0)

    @property
    def dataset_info(self):
        return "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification", 1241.20, "Data"
      
    @scenario(name="GTZAN CVAE", version="1.0-LOW16", description="Scenarios in this series aim to find a decent latent space size for low quality audio data more diverse than simple sinewave compounds. 6 convolutional layers with symmetrical stride are used.\nLatent Space Size : 16 dimensions")
    def cvae_core16(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[ 32, 64, 128, 256, 512, 1024],
            conv_kernels=[  3,  3,   3,   3,   3,    3],
            conv_strides=[  2,  2,   2,   2,   2,    2],
            latent_space_dim=16
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer

    @scenario(name="GTZAN CVAE", version="1.0-LOW32", description="Scenarios in this series aim to find a decent latent space size for low quality audio data more diverse than simple sinewave compounds. 6 convolutional layers with symmetrical stride are used.\nLatent Space Size : 32 dimensions")
    def cvae_core32(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[ 32, 64, 128, 256, 512, 1024],
            conv_kernels=[  3,  3,   3,   3,   3,    3],
            conv_strides=[  2,  2,   2,   2,   2,    2],
            latent_space_dim=32
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="GTZAN CVAE", version="1.0-LOW64", description="Scenarios in this series aim to find a decent latent space size for low quality audio data more diverse than simple sinewave compounds. 6 convolutional layers with symmetrical stride are used.\nLatent Space Size : 64 dimensions")
    def cvae_core64(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[ 32, 64, 128, 256, 512, 1024],
            conv_kernels=[  3,  3,   3,   3,   3,    3],
            conv_strides=[  2,  2,   2,   2,   2,    2],
            latent_space_dim=64
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
