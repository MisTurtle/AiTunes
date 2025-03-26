from typing import Iterable
import numpy as np
import torch.optim as optim

from os import path

from aitunes.modules import CVAE, ResNet2dV1, VQ_ResNet2D
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
        return HighResolutionAudioFeatures(10.0), LowResolutionAudioFeatures(10.0)

    @property
    def dataset_info(self):
        return None
    
    @property
    def duration(self) -> float:
        return 10.0
    
    def _create_audios(self):
        generate_dataset_of_simple_instruments(self.path_to_audio_root, sample_rate=self.all_modes[0].sample_rate, unit_duration=self.duration, unit_per_type=500)
    
    def map_filename_to_label(self, filename):
        if not isinstance(filename, Iterable):
            filename = [filename]
        elif isinstance(filename, np.ndarray):
            filename = map(lambda byteFilename: byteFilename.decode('utf-8'), filename.tolist())
        return [f.split('.')[0] for f in filename]

    @scenario(name="CVAE", version="low-dim8", description="Scenarios in this series aim to find a decent latent space size for low quality audio data as simple as sinewave combinations. Latent Dim: 8")
    def cvae_low8(self):
        self.mode = 1
        model = CVAE(
            input_shape=(1, *self.mode.spectrogram_size),
            conv_channels=(  32,   64,  128,  256,  256),
            conv_kernels=(   3,    3,     3,    3,    1),
            conv_strides=(   2,    2,     2,    2,    1),
            latent_dimension=8
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        return model, loss, optimizer

    @scenario(name="CVAE", version="low-dim16", description="Scenarios in this series aim to find a decent latent space size for low quality audio data as simple as sinewave combinations. Latent Dim: 16")
    def cvae_low16(self):
        self.mode = 1
        model = CVAE(
            input_shape=(1, *self.mode.spectrogram_size),
            conv_channels=(  32,   64,  128,  256,  256),
            conv_kernels=(   3,    3,     3,    3,    1),
            conv_strides=(   2,    2,     2,    2,    1),
            latent_dimension=16
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        return model, loss, optimizer

    @scenario(name="CVAE", version="low-dim32", description="Scenarios in this series aim to find a decent latent space size for low quality audio data as simple as sinewave combinations. Latent Dim: 32")
    def cvae_low32(self):
        self.mode = 1
        model = CVAE(
            input_shape=(1, *self.mode.spectrogram_size),
            conv_channels=(  32,   64,  128,  256,  256),
            conv_kernels=(   3,    3,     3,    3,    1),
            conv_strides=(   2,    2,     2,    2,    1),
            latent_dimension=32
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        return model, loss, optimizer

    @scenario(name="ResNet2D", version="low-dim32", description="Application of the residual network infrastructure on audio data. Results were pretty good with the CIFAR10 experiment, so this scenario will attempt to validate its superiority. Latent Dim: 32")
    def resnet_low32(self):
        self.mode = 1
        model = ResNet2dV1((1, *self.mode.spectrogram_size), 4, 16, 32)
        
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        return model, loss, optimizer
    
    @scenario(name="VQ-ResNet2D", version="v1", description="An implementation of the VQ-VAE model using a ResNet2D convolutional architecture. This test takes the first VQVAE Jukebox layer parameters and directly uncompresses the spectrogram back to its original size")
    def vq_resnet_test1(self):
        self.mode = 1
        model = VQ_ResNet2D(
            input_shape=(1, *self.mode.spectrogram_size),
            num_hiddens=256,
            num_downsampling_layers=4,
            num_residual_layers=5,
            num_residual_hiddens=128,
            embedding_dim=64,
            num_embeddings=2048,
            use_ema=True,
            random_restart=512
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),  # Reconstruction loss
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1)  # Codebook loss
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

