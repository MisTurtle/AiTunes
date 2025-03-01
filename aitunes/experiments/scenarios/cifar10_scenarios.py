import torch.nn as nn
import torch.optim as optim

from os import path
from aitunes.experiments.cases import Cifar10Experiment
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario
from aitunes.utils.loss_functions import *
from aitunes.modules import CVAE, ResNet2D, VQ_ResNet2D


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
        return "The CIFAR10 dataset consists of 60000 3x32x32 images representing color images from ten different categories. This implementation aims to expand on the simple MNIST dataset with more complex images and see the generative potential of VAE architectures."
        
    def instantiate(self, s, model_path):
        model, loss, optimizer = s(self)
        return Cifar10Experiment(model, model_path or s.model_path, loss, optimizer)
    
    @scenario(name="CVAE", version="dim16", description="This series aims to set a base and visually compare results obtained with different latent dimensions. It uses MSE Loss combined with KL Divergence loss linearly annealed over 10 epochs. Latent Dim: 16")
    def cvae_core16(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256, 256],
            conv_kernels=[ 3,  3,   3,   3,   3],
            conv_strides=[ 2,  2,   2,   2,   1],
            latent_space_dim=16
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.000125)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="dim64", description="This series aims to set a base and visually compare results obtained with different latent dimensions. It uses MSE Loss combined with KL Divergence loss linearly annealed over 10 epochs. Latent Dim: 64")
    def cvae_core64(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256, 256],
            conv_kernels=[ 3,  3,   3,   3,   3],
            conv_strides=[ 2,  2,   2,   2,   1],
            latent_space_dim=64
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.00003125)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="dim256", description="This series aims to set a base and visually compare results obtained with different latent dimensions. It uses MSE Loss combined with KL Divergence loss linearly annealed over 10 epochs. Latent Dim: 256")
    def cvae_core256(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256, 256],
            conv_kernels=[ 3,  3,   3,   3,   3],
            conv_strides=[ 2,  2,   2,   2,   1],
            latent_space_dim=256
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.0000078125)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer

    @scenario(name="CVAE", version="ssim256", description="This scenario aims to visually compare results obtained when replacing the MSE Loss with a Structural Similarity loss, focusing on perception rather than pixelwise loss. If results end up looking better, a similar approach could be used with audio scenarios in other experiments. Latent Dim: 256")
    def cvae_ssim256(self):
        model = CVAE(
            input_shape =[ 3, 32,  32],
            conv_filters=[32, 64, 128, 256, 256],
            conv_kernels=[ 3,  3,   3,   3,   3],
            conv_strides=[ 2,  2,   2,   2,   1],
            latent_space_dim=256
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.0000078125),
            (create_ssim_loss_function(win_size=5), 0.05)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
    
    @scenario(name="ResNet2D", version="dim256", description="Attempt at improving the simple CVAE structure we've been using so far with a deeper, more complex ResNet 2D model. Latent Dim: 256")
    def resnet256(self):
        model = ResNet2D((3, 32, 32), 3, 64, 256)
        
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.0000078125),
            (create_ssim_loss_function(win_size=5), 0.05)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss, optimizer
        
    @scenario(name="VQ-ResNet2D", version="dim64", description="An attempt at implementing a simple VQ-VAE model (using the ResNet2D convolutional architecture). Latent Dim: 32")
    def vq_resnet_low64(self):
        self.mode = 1
        model = VQ_ResNet2D(
            input_shape=(3, 32, 32),
            num_hiddens=128,
            num_downsampling_layers=2,
            num_residual_layers=2,
            num_residual_hiddens=32,
            embedding_dim=64,
            num_embeddings=512
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        return model, loss, optimizer
