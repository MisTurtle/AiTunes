import torch.optim as optim

from os import path
from aitunes.experiments.cases import Cifar10Experiment
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario
from aitunes.utils.loss_functions import simple_mse_kl_loss
from aitunes.modules import CVAE


class Cifar10ReconstructionScenarios(ScenarioContainer):

    @property
    def path_to_history_root(self):
        return path.join("history", "cifar10")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "cifar10")

    @property
    def identifier(self):
        return "CIFAR 10"

    @property
    def description(self):
        return "The CIFAR10 dataset consists of 60000 3x32x32 images representing colour images from ten different categories. This implementation aims to expand on the simple MNIST dataset with more complex images and see the generative potential of VAE architectures."
        
    def instantiate(self, s, model_path):
        model, loss, optimizer = s(self)
        return Cifar10Experiment(model, model_path or s.model_path, loss, optimizer)
    
    @scenario(name="Simple CVAE", version="1.0", description="Simple VAE implementation using convolutionnal layers. This scenario aims to set a base and visually compare to results obtained with the VQ-VAE infrastructure. Latent Dim: 8")
    def cvae(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256],
            conv_kernels=[ 3,  3,   3,   3],
            conv_strides=[ 2,  2,   2,   2],
            latent_space_dim=8
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple CVAE", version="1.0-CORE32", description="Simple VAE implementation using convolutionnal layers. This scenario aims to set a base and visually compare to results obtained with the VQ-VAE infrastructure. Latent Dim: 32")
    def cvae_core32(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256],
            conv_kernels=[ 3,  3,   3,   3],
            conv_strides=[ 2,  2,   2,   2],
            latent_space_dim=32
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer

    @scenario(name="Simple CVAE", version="1.0-CORE64", description="Simple VAE implementation using convolutionnal layers. This scenario aims to set a base and visually compare to results obtained with the VQ-VAE infrastructure. Latent Dim: 64")
    def cvae_core64(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256],
            conv_kernels=[ 3,  3,   3,   3],
            conv_strides=[ 2,  2,   2,   2],
            latent_space_dim=64
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple CVAE", version="1.0-CORE128", description="Simple VAE implementation using convolutionnal layers. This scenario aims to set a base and visually compare to results obtained with the VQ-VAE infrastructure. Latent Dim: 128")
    def cvae_core128(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256],
            conv_kernels=[ 3,  3,   3,   3],
            conv_strides=[ 2,  2,   2,   2],
            latent_space_dim=128
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple CVAE", version="1.0-CORE128-REC", description="Simple VAE implementation using convolutionnal layers. This scenario aims to set a base and visually compare to results obtained with the VQ-VAE infrastructure. Latent Dim: 128")
    def cvae_core128_rec(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256],
            conv_kernels=[ 3,  3,   3,   3],
            conv_strides=[ 2,  2,   2,   2],
            latent_space_dim=128
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=0.1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer