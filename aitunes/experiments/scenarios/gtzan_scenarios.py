from os import path
import torch.optim as optim

from aitunes.modules import CVAE, ResNet2dV1
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
from aitunes.modules.autoencoder_modules import VQ_ResNet2D
from aitunes.utils.loss_functions import *
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
        return HighResolutionAudioFeatures(10.0), LowResolutionAudioFeatures(10.0)

    @property
    def dataset_info(self):
        return "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification", 1241.20, "Data"

    @scenario(name="ResNet2D", version="low-dim16", description="Application of the residual network architecture on complexe, low-quality audio data. Latent Dim: 16")
    def resnet_low16(self):
        self.mode = 1
        model = ResNet2dV1((1, *self.mode.spectrogram_size), 4, 32, 16)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 32)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

    @scenario(name="ResNet2D", version="low-dim32", description="Application of the residual network architecture on complexe, low-quality audio data. Latent Dim: 32")
    def resnet_low32(self):
        self.mode = 1
        model = ResNet2dV1((1, *self.mode.spectrogram_size), 4, 32, 32)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 32)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="ResNet2D", version="low-dim64", description="Application of the residual network architecture on complexe, low-quality audio data. Latent Dim: 64")
    def resnet_low64(self):
        self.mode = 1
        model = ResNet2dV1((1, *self.mode.spectrogram_size), 4, 64, 64)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 32)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="ResNet2D", version="low-dim128", description="Application of the residual network architecture on complexe, low-quality audio data. Latent Dim: 128")
    def resnet_low128(self):
        self.mode = 1
        model = ResNet2dV1((1, *self.mode.spectrogram_size), 4, 64, 128)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 32)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="ResNet2D", version="low-dim256", description="Application of the residual network architecture on complexe, low-quality audio data. Latent Dim: 256")
    def resnet_low256(self):
        self.mode = 1
        model = ResNet2dV1((1, *self.mode.spectrogram_size), 5, 64, 256)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 32)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

    @scenario(name="VQ-ResNet2D", version="test1", description="An attempt at implementing a simple VQ-VAE model (using a better ResNet2D convolutional architecture). Latent Dim: 32, Discrete Vectors: 512, Downsampling: 2")
    def vq_resnet_test1(self):
        self.mode = 1
        model = VQ_ResNet2D(
            input_shape=(1, *self.mode.spectrogram_size),
            num_hiddens=128,
            num_downsampling_layers=2,
            num_residual_layers=3,
            num_residual_hiddens=64,
            embedding_dim=32,
            num_embeddings=512,
            use_ema=True,
            random_restart=16
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),  # Reconstruction loss
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1),  # Codebook loss
            # (create_cherry_picked_loss((2, ), (1, )), 1)  # Perplexity
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

    @scenario(name="VQ-ResNet2D", version="test2", description="An attempt at implementing a simple VQ-VAE model (using a better ResNet2D convolutional architecture). Latent Dim: 128, Discrete Vectors: 8192, Downsampling: 2")
    def vq_resnet_test2(self):
        self.mode = 1
        model = VQ_ResNet2D(
            input_shape=(1, *self.mode.spectrogram_size),
            num_hiddens=256,
            num_downsampling_layers=2,
            num_residual_layers=3,
            num_residual_hiddens=128,
            embedding_dim=128,
            num_embeddings=8192,
            use_ema=True,
            random_restart=64
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),  # Reconstruction loss
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1),  # Codebook loss
            # (create_cherry_picked_loss((2, ), (1, )), 1)  # Perplexity
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="VQ-ResNet2D", version="test3", description="An attempt at implementing a simple VQ-VAE model (using a better ResNet2D convolutional architecture). Latent Dim: 128, Discrete Vectors: 8192")
    def vq_resnet_test3(self):
        self.mode = 1
        model = VQ_ResNet2D(
            input_shape=(1, *self.mode.spectrogram_size),
            num_hiddens=256,
            num_downsampling_layers=3,
            num_residual_layers=3,
            num_residual_hiddens=128,
            embedding_dim=128,
            num_embeddings=8192,
            use_ema=True,
            random_restart=64
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),  # Reconstruction loss
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1),  # Codebook loss
            # (create_cherry_picked_loss((2, ), (1, )), 1)  # Perplexity
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
