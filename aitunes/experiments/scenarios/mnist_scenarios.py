import torch.optim as optim

from os import path
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario
from aitunes.utils.loss_functions import mse_loss, simple_mse_kl_loss
from aitunes.experiments.cases import MnistExperiment
from aitunes.modules import SimpleAutoEncoder, VariationalAutoEncoder, CVAE


class MnistReconstructionScenarios(ScenarioContainer):

    @property
    def path_to_history_root(self):
        return path.join("history", "mnist")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "mnist")

    @property
    def identifier(self):
        return "MNIST"

    @property
    def description(self):
        return "The MNIST dataset consists of 28x28 images representing digits from 0 to 9. This experiment aims to showcase the limitations of simple architectures and how more advanced ones like CVAE can be used for generating meaningful images from random latent samples."
        
    def instantiate(self, s, model_path):
        model, loss, optimizer = s(self)
        return MnistExperiment(model, model_path or s.model_path, loss, optimizer)
    
    @scenario(name="Simple AE", version="1.0", description="Train to reconstruct images from compression to a 2D plane with a simple autoencoder")
    def ae(self):
        model = SimpleAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss, optimizer = mse_loss, optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple VAE", version="1.0", description="Hint the model to follow a standard random normal distribution for compressing images to a 2D plane thanks to a KL Divergence loss. This allows to easily sample new points from the latent space. Reconstruction loss and KL divergence loss are balanced in weights.")
    def vae(self):
        model = VariationalAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple VAE", version="1.0-KL", description="Compared to 1.0, this model gives a higher importance to the KL Divergence loss. This aims to compare the results when multiplying its weight by 10")
    def vae_kl(self):
        model = VariationalAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=10), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple VAE", version="1.0-REC", description="Compared to 1.0, this model gives a higher importance to the Reconstruction loss. This aims to compare the results when multiplying its weight by 10")
    def vae_rec(self):
        model = VariationalAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=0.1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple CVAE", version="1.0", description="Use convolutions to extract meaningful features through relations between clusters of pixels. Also serves as a test to verify the CVAE implementation work as expected. Uses 3 convolutional layers as well as balanced weights due to better results being yielded by this setting for the Simple VAE model")
    def cvae(self):
        model = CVAE(
            input_shape =[ 1, 28,  28],
            conv_filters=[32, 64, 128],
            conv_kernels=[ 3,  3,   3],
            conv_strides=[ 2,  2,   2],
            latent_space_dim=2
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="Simple CVAE", version="1.0-REC", description="Use convolutions to extract meaningful features through relations between clusters of pixels. Also serves as a test to verify the CVAE implementation work as expected. Uses 3 convolutional layers as well as balanced weights due to better results being yielded by this setting for the Simple VAE model. Beta = 0.1")
    def cvae_rec(self):
        model = CVAE(
            input_shape =[ 1, 28,  28],
            conv_filters=[32, 64, 128],
            conv_kernels=[ 3,  3,   3],
            conv_strides=[ 2,  2,   2],
            latent_space_dim=2
        )
        loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=0.1), optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    