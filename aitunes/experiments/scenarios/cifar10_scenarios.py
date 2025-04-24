import torch.nn as nn
import torch.optim as optim

from os import path
from aitunes.experiments.cases import Cifar10Experiment
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario
from aitunes.utils.loss_functions import *
from aitunes.modules import CVAE, ResNet2dV1, VQ_ResNet2D


class Cifar10ReconstructionScenarios(ScenarioContainer):

    @property
    def path_to_history_root(self):
        return path.join("history", "cifar10")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "cifar10")

    @property
    def identifier(self):
        return "CIFAR10"

    @property
    def description(self):
        return "The CIFAR10 dataset consists of 60000 3x32x32 images representing color images from ten different categories. This implementation aims to expand on the simple MNIST dataset with more complex images and see the generative potential of VAE architectures."
        
    def instantiate(self, s, model_path, light_mode: bool = False):
        model, loss, optimizer = s(self)
        return Cifar10Experiment(model, model_path or s.model_path, loss, optimizer)  # TODO : Light mode support
    
    @scenario(name="CVAE", version="dim16", description="This series aims to set a base and visually compare results obtained with different latent dimensions. It uses MSE Loss combined with KL Divergence loss linearly annealed over 5 epochs. Latent Dim: 16")
    def cvae_core16(self):
        model = CVAE(
            input_shape=  (3, 32, 32),
            conv_channels=( 32,  32,  64,  64,  128,  128,  256,  256),
            conv_kernels= (  3,   3,   3,   3,    3,    3,    3,    3),
            conv_strides= (  2,   1,   2,   1,    2,    1,    2,    1),
            latent_dimension=16
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 16)), 0.0001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="dim32", description="This series aims to set a base and visually compare results obtained with different latent dimensions. It uses MSE Loss combined with KL Divergence loss linearly annealed over 5 epochs. Latent Dim: 32")
    def cvae_core32(self):
        model = CVAE(
            input_shape=  (3, 32, 32),
            conv_channels=( 32,  32,  64,  64,  128,  128,  256,  256),
            conv_kernels= (  3,   3,   3,   3,    3,    3,    3,    3),
            conv_strides= (  2,   1,   2,   1,    2,    1,    2,    1),
            latent_dimension=32
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 16)), 0.0001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="dim64", description="This series aims to set a base and visually compare results obtained with different latent dimensions. It uses MSE Loss combined with KL Divergence loss linearly annealed over 5 epochs. Latent Dim: 64")
    def cvae_core64(self):
        model = CVAE(
            input_shape=  (3, 32, 32),
            conv_channels=( 32,  32,  64,  64,  128,  128,  256,  256),
            conv_kernels= (  3,   3,   3,   3,    3,    3,    3,    3),
            conv_strides= (  2,   1,   2,   1,    2,    1,    2,    1),
            latent_dimension=64
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 16)), 0.0001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="dim128", description="This series aims to set a base and visually compare results obtained with different latent dimensions. It uses MSE Loss combined with KL Divergence loss linearly annealed over 5 epochs. Latent Dim: 128")
    def cvae_core128(self):
        model = CVAE(
            input_shape=  (3, 32, 32),
            conv_channels=( 32,  32,  64,  64,  128,  128,  256,  256),
            conv_kernels= (  3,   3,   3,   3,    3,    3,    3,    3),
            conv_strides= (  2,   1,   2,   1,    2,    1,    2,    1),
            latent_dimension=128
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 16)), 0.0001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="ssim128", description="This scenario aims to visually compare results obtained when adding a Structural Similarity loss, focusing on perception rather than pixelwise loss. If results end up looking better, a similar approach could be used with audio scenarios in other experiments. Latent Dim: 128")
    def cvae_ssim128(self):
        model = CVAE(
            input_shape=  (3, 32, 32),
            conv_channels=( 32,  32,  64,  64,  128,  128,  256,  256),
            conv_kernels= (  3,   3,   3,   3,    3,    3,    3,    3),
            conv_strides= (  2,   1,   2,   1,    2,    1,    2,    1),
            latent_dimension=128
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 16)), 0.0001),
            (create_ssim_loss_function(win_size=5), 0.05)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="ResNet2D", version="hello_world", description="Attempt at improving the simple CVAE structure we've been using so far with a deeper, more complex ResNet 2D model. Latent Dim: 128")
    def resnet_hello_world(self):
        model = ResNet2dV1((3, 32, 32), 4, 32, 128)
        
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 16)), 0.0001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
        
    @scenario(name="VQ-ResNet2D", version="hello_world", description="An attempt at implementing a simple VQ-VAE model (using a better ResNet2D convolutional architecture). Latent Dim: 64, Discrete Vectors: 512")
    def vq_resnet_hello_world(self):
        self.mode = 1
        model = VQ_ResNet2D(
            input_shape=(3, 32, 32),
            num_hiddens=128,
            num_downsampling_layers=4,
            num_residual_layers=5,
            num_residual_hiddens=64,
            embedding_dim=32,
            num_embeddings=2048,
            use_ema=True,
            random_restart=16
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),  # Reconstruction loss
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1),  # Codebook loss
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
        
    @scenario(name="VQ-ResNet2D", version="better_world", description="An attempt at implementing a simple VQ-VAE model (using a better ResNet2D convolutional architecture). Latent Dim: 32, Discrete Vectors: 2048")
    def vq_resnet_better_world(self):
        self.mode = 1
        model = VQ_ResNet2D(
            input_shape=(3, 32, 32),
            num_hiddens=128,
            num_downsampling_layers=2,
            num_residual_layers=3,
            num_residual_hiddens=64,
            embedding_dim=32,
            num_embeddings=2048,
            use_ema=True,
            random_restart=16
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),  # Reconstruction loss
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1),  # Codebook loss
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
