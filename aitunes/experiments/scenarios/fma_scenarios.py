import torch.optim as optim

from os import path

from aitunes.experiments.scenarios._scenario_utils import AudioBasedScenarioContainer, scenario
from aitunes.modules.autoencoder_modules import CVAE
from aitunes.utils.audio_utils import HighResolutionAudioFeatures, LowResolutionAudioFeatures
from aitunes.utils.loss_functions import simple_mse_kl_loss


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
        return HighResolutionAudioFeatures(5.0), LowResolutionAudioFeatures(5.0)

    @property
    def dataset_info(self) -> tuple:
        return "https://os.unil.cloud.switch.ch/fma/fma_medium.zip", 22721.29, "fma_medium"

    @scenario(name="FMA CVAE", version="1.0-LOW32", description="")
    def cvae_core32(self):
        self.mode = 1
        model = CVAE(
            input_shape=[1, *self.mode.spectrogram_size],
            conv_filters=[ 32, 64, 128, 256, 512, 1024],
            conv_kernels=[  3,  3,   3,   3,   3,    3],
            conv_strides=[  (1, 2),  (1, 2),   (1, 2),   2,   2,    2],
            latent_space_dim=64
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
