from typing import Iterable
import torch.optim as optim

from os import path

from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
from aitunes.modules.autoencoder_modules import CVAE, ResNet2dV1, VQ_ResNet2D
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures
from aitunes.utils.loss_functions import combine_losses, create_cherry_picked_loss, create_kl_loss_with_linear_annealing, create_mse_loss, simple_mse_kl_loss


class FmaReconstructionScenarios(AudioBasedScenarioContainer):

    @property
    def path_to_history_root(self):
        return path.join("history", "fma")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "fma")

    @property
    def identifier(self):
        return "FMA"

    @property
    def description(self):
        return "A significantly larger dataset for training on music of various types. Due to its size, only production-grade models will be trained on this dataset."

    @property
    def path_to_dataset_root(self):
        return path.join("assets", "Samples", "FMA")

    @property
    def all_paths_to_training_spectrograms(self):
        return path.join(self.path_to_dataset_root, "spectrograms", "training_high.h5"), path.join(self.path_to_dataset_root, "spectrograms", "training_low.h5")

    @property
    def all_paths_to_eval_spectrograms(self):
        return path.join(self.path_to_dataset_root, "spectrograms", "eval_high.h5"), path.join(self.path_to_dataset_root, "spectrograms", "eval_low.h5")

    @property
    def path_to_audio_root(self):
        return path.join(self.path_to_dataset_root, "fma_medium")

    @property
    def all_modes(self):
        return HighResolutionAudioFeatures(10.0), LowResolutionAudioFeatures(10.0)

    @property
    def dataset_info(self) -> tuple:
        return "https://os.unil.cloud.switch.ch/fma/fma_medium.zip", 22721.29, "fma_medium"

    def map_filename_to_label(self, filename):
        print(isinstance(filename, Iterable))
        return filename
    
    @scenario(name="ResNet2D", version="high-dim64", description="Application of the residual network architecture on complexe, high-quality audio data, with a fix for BatchNorm2d layers breaking during evaluation due to them behaving differently with different modes. Latent Dim: 64")
    def resnet_high64(self):
        self.mode = 0
        model = ResNet2dV1((1, *self.mode.spectrogram_size), 4, 64, 64, bn_momentum=0.001)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 32)), 0.00001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        return model, loss, optimizer

    @scenario(name="VQ-ResNet2D", version="v1", description="An implementation of the VQ-VAE model using a ResNet2D convolutional architecture. This test takes the first VQVAE Jukebox layer parameters and directly uncompresses the spectrogram back to its original size")
    def vq_resnet_test1(self):
        self.mode = 0
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
