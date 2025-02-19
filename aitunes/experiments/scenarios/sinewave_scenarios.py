import torch.optim as optim

from os import path

from aitunes.modules import CVAE
from aitunes.audio_generation.simple_audio_streams import generate_dataset_of_simple_instruments
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures
from aitunes.utils.loss_functions import simple_mse_kl_loss


class SinewaveReconstructionScenarios(AudioBasedScenarioContainer):

    @property
    def path_to_history_root(self):
        return path.join("history", "sinewave")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "sinewave")

    @property
    def identifier(self):
        return "SINEWAVE"

    @property
    def description(self):
        return "Spectrograms of simple sine compound waves are attempted to be reconstructed after being compressed. This experiment aims to find a decent working architecture for audio creation to start tests on the GTZAN dataset which contains actual music."

    @property
    def path_to_dataset_root(self):
        return path.join("assets", "Samples", "SINEWAVE")

    @property
    def all_paths_to_training_spectrograms(self):
        return path.join(self.path_to_dataset_root, "spectrograms", "training_high.h5"), path.join(self.path_to_dataset_root, "spectrograms", "training_low.h5")

    @property
    def all_paths_to_eval_spectrograms(self):
        return path.join(self.path_to_dataset_root, "spectrograms", "eval_high.h5"), path.join(self.path_to_dataset_root, "spectrograms", "eval_low.h5")

    @property
    def path_to_audio_root(self):
        return path.join(self.path_to_dataset_root, "audio")

    @property
    def all_modes(self):
        return HighResolutionAudioFeatures(5.0), LowResolutionAudioFeatures(5.0)

    @property
    def dataset_info(self):
        return None
    
    def _create_audios(self):
        generate_dataset_of_simple_instruments(self.path_to_audio_root, sample_rate=self.all_modes[0].sample_rate, unit_duration=5.0, unit_per_type=100)
        
    @scenario(name="Simple CVAE", version="1.0-LOW4", description="Scenarios in this series aim to find a decent latent space size for low quality audio data. 3 convolutional layers with symmetrical strides are used.\nLatent Space Size : 4 dimensions")
    def cvae_core4(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[  32,   64,  128],
            conv_kernels=[   3,    3,    3],
            conv_strides=[   2,    2,    2],
            latent_space_dim=4
        )
        loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer

    @scenario(name="Simple CVAE", version="1.0-LOW8", description="Scenarios in this series aim to find a decent latent space size for low quality audio data. 3 convolutional layers with symmetrical stride are used.\nLatent Space Size : 8 dimensions")
    def cvae_core8(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[  32,   64,  128],
            conv_kernels=[   3,    3,    3],
            conv_strides=[   2,    2,    2],
            latent_space_dim=8
        )
        loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple CVAE", version="1.0-LOW16", description="Scenarios in this series aim to find a decent latent space size for low quality audio data. 3 convolutional layers with symmetrical stride are used.\nLatent Space Size : 16 dimensions")
    def cvae_core16(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[  32,   64,  128],
            conv_kernels=[   3,    3,    3],
            conv_strides=[   2,    2,    2],
            latent_space_dim=16
        )
        loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
        