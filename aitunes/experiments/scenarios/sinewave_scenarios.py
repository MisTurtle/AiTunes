import torch.optim as optim

from os import path

from aitunes.modules import CVAE, ResNet2D, VQ_ResNet2D
from aitunes.audio_generation.simple_audio_streams import generate_dataset_of_simple_instruments
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures
from aitunes.utils.loss_functions import *


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
        generate_dataset_of_simple_instruments(self.path_to_audio_root, sample_rate=self.all_modes[0].sample_rate, unit_duration=5.0, unit_per_type=500)
        
    @scenario(name="CVAE", version="low-dim2", description="Scenarios in this series aim to find a decent latent space size for low quality audio data as simple as sinewave combinations. Latent Dim: 2")
    def cvae_low2(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[  32,   64,  128,  256,  256],
            conv_kernels=[   3,    3,    3,    3,    1],
            conv_strides=[   2,    2,    2,    2,    1],
            latent_space_dim=2
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

    @scenario(name="CVAE", version="low-dim16", description="Scenarios in this series aim to find a decent latent space size for low quality audio data as simple as sinewave combinations. Latent Dim: 16")
    def cvae_low16(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[  32,   64,  128,  256,  256],
            conv_kernels=[   3,    3,    3,    3,    1],
            conv_strides=[   2,    2,    2,    2,    1],
            latent_space_dim=16
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.000125)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

    @scenario(name="CVAE", version="low-dim32", description="Scenarios in this series aim to find a decent latent space size for low quality audio data as simple as sinewave combinations. Latent Dim: 32")
    def cvae_low32(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[  32,   64,  128,  256,  256],
            conv_kernels=[   3,    3,    3,    3,    1],
            conv_strides=[   2,    2,    2,    2,    1],
            latent_space_dim=32
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00003125)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

    @scenario(name="ResNet2D", version="low-dim32", description="Application of the residual network infrastructure on audio data. Results were pretty good with the CIFAR10 experiment, so this scenario will attempt to validate its superiority. Latent Dim: 32")
    def resnet_low32(self):
        self.mode = 1
        model = ResNet2D((1, *self.mode.spectrogram_size), 4, 16, 32)
        
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00003125)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="VQ-ResNet2D", version="low-dim32", description="An attempt at implementign a simple VQ-VAE model (using the ResNet2D convolutional architecture). Latent Dim: 32")
    def vq_resnet_low32(self):
        self.mode = 1
        model = VQ_ResNet2D((1, *self.mode.spectrogram_size), 4, 16, 32, 8192)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

