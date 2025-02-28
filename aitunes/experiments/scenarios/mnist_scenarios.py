import torch.optim as optim

from os import path
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario
from aitunes.utils.loss_functions import *
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
    
    @scenario(name="AE", version="mse", description="Train to reconstruct images from compression to a 2D plane with a simple autoencoder")
    def ae(self):
        model = SimpleAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss = create_mse_loss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="default", description="Hint the model to follow a standard normal distribution for compressing images to a 2D plane using a KL Divergence loss combined with MSE loss for pixel-wise error measurement. Points from the latent space can be sampled from R2. KL divergence weight = 0.001")
    def vae(self):
        model = VariationalAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss(reduce=True), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="kullback", description="Compared to the default version, this scenario gives a higher importance to the KL Divergence loss. KL divergence weight = 1")
    def vae_kullback(self):
        model = VariationalAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss(reduce=True), 1)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="lin_annealing", description="This scenario gives more importance to the Reconstruction loss at the beginning and ends up with a default loss after around 5 epochs. KL divergence weight = [0 to 0.001]")
    def vae_linear_annealing(self):
        model = VariationalAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 32)), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="cyc_annealing", description="This scenario tries to show benefits from having a cyclic KL Divergence loss weight, going from 0 to 0.002 over X=2 epochs and repeating. This is dangerous though as we don't really know how many epochs the user is going to train the model on, so it probably should always be a multiple of X and have a very close estimation for the # of batch per epoch... A better approach should definitely be tacled")
    def vae_cyclic_annealing(self):
        model = VariationalAutoEncoder((28 * 28, 7 * 7, 3 * 3, 2))
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_cyclic_annealing(over_epochs=2, batch_per_epoch=int(50000 / 32)), 0.002)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="default", description="Use convolutions to extract meaningful features through relations between clusters of pixels. Also serves as a test to verify the CVAE implementation work as expected. Uses 3 convolutional layers as well as linear KL Loss annealing due to better results being yielded by this method.")
    def cvae(self):
        model = CVAE(
            input_shape =[ 1, 28,  28],
            conv_filters=[32, 64, 128],
            conv_kernels=[ 3,  3,   3],
            conv_strides=[ 2,  2,   2],
            latent_space_dim=2
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 32)), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
