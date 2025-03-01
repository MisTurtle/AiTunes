from os import path
import torch.optim as optim

from aitunes.modules import CVAE, ResNet2D
from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
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
        return HighResolutionAudioFeatures(2.0), LowResolutionAudioFeatures(2.0)

    @property
    def dataset_info(self):
        return "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification", 1241.20, "Data"
      
    @scenario(name="ResNet2D", version="low-dim32", description="Application of the residual network architecture on complexe, low-quality audio data. Latent Dim: 32")
    def resnet_low32(self):
        self.mode = 1
        print(self.mode.spectrogram_size)
        model = ResNet2D((1, *self.mode.spectrogram_size), 4, 32, 32)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.0000625)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="ResNet2D", version="low-dim256", description="Application of the residual network architecture on complexe, low-quality audio data. Latent Dim: 256")
    def resnet_low256(self):
        self.mode = 1
        print(self.mode.spectrogram_size)
        model = ResNet2D((1, *self.mode.spectrogram_size), 4, 32, 256)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.0000078125)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="ResNet2D", version="high-dim512", description="Application of the residual network architecture on complexe, high-quality audio data. Latent Dim: 512")
    def resnet_high512(self):
        self.mode = 0
        print(self.mode.spectrogram_size)
        model = ResNet2D((1, *self.mode.spectrogram_size), 6, 32, 512)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00000390625)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer

    @scenario(name="ResNet2D", version="high-dim1024", description="Application of the residual network architecture on complexe, high-quality audio data. Latent Dim: 1024")
    def resnet_high1024(self):
        self.mode = 0
        print(self.mode.spectrogram_size)
        model = ResNet2D((1, *self.mode.spectrogram_size), 6, 32, 1024)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00000390625)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    