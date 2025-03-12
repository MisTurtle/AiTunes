import torch.optim as optim

from os import path
from aitunes.modules import *
from aitunes.utils.loss_functions import *
from aitunes.experiments.cases import MnistExperiment
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario


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
    
    @scenario(name="AE", version="vanilla", description="Train to reconstruct images from compression to a 2D plane with a vanilla autoencoder.")
    def ae_vanilla(self):
        model = VanillaAutoEncoder(1 * 28 * 28, [32 * 32, 16 * 16, 4 * 4], 2)
        loss = create_mse_loss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="small_kl", description="Hint the model to follow a standard normal distribution for compressing images to a 2D plane using a KL Divergence loss combined with MSE loss for pixel-wise error measurement. Points from the latent space can be sampled from R2. KL divergence weight = 0.001")
    def vae_small_kl(self):
        model = VariationalAutoEncoder(1 * 28 * 28, [32 * 32, 16 * 16, 4 * 4], 2)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss(reduce=True), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="big_kl", description="Compared to the first version, this scenario gives a higher importance to the KL Divergence loss. KL divergence weight = 0.01")
    def vae_big_kl(self):
        model = VariationalAutoEncoder(1 * 28 * 28, [32 * 32, 16 * 16, 4 * 4], 2)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss(reduce=True), 0.01)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="linear_kl", description="This scenario gives more importance to the Reconstruction loss at the beginning and ends up with a kl loss weight of 0.001 after around 5 epochs. KL divergence weight = [0 to 0.001]")
    def vae_linear_annealing(self):
        model = VariationalAutoEncoder(1 * 28 * 28, [32 * 32, 16 * 16, 4 * 4], 2)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 32)), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="cyclic_kl", description="This scenario tries to show benefits raised at https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/ regarding cyclic KL Divergence loss weights, going from 0 to 0.001 over X=2 epochs and repeating. This might be dangerous though as we don't really know how many epochs the user is going to train the model on, so it probably should always be a multiple of X and have a very close estimation for the # of batch per epoch... KL divergence weight = [0 to 0.001 to 0...]")
    def vae_cyclic_annealing(self):
        model = VariationalAutoEncoder(1 * 28 * 28, [32 * 32, 16 * 16, 4 * 4], 2)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_cyclic_annealing(over_epochs=2, batch_per_epoch=int(50000 / 32)), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="CVAE", version="hello_world", description="Use convolutions to force the model to learn relations between clusters of pixels. Also serves as a test for the basic CVAE architecture. Linear KL Loss annealing over 5 epochs is used. KL divergence weight: 0.001")
    def cvae_hello_world(self):
        model = CVAE(
            input_shape=  (1, 28, 28),
            conv_channels=( 32,  32,  64,  64,  128,  128),
            conv_kernels= (  3,   3,   3,   3,    3,    3),
            conv_strides= (  2,   1,   2,   1,    2,    1),
            latent_dimension=2
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss_with_linear_annealing(over_epochs=5, batch_per_epoch=int(50000 / 32)), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer

    @scenario(name="VQVAE", version="hello_world", description="A test of the VQ VAE infrastructure on the easy case of MNIST digits reconstruction.")
    def vqvae_hello_world(self):
        model = VQ_ResNet2D(
            input_shape=(1, 28, 28),
            num_hiddens=64,
            num_downsampling_layers=3,
            num_residual_layers=3,
            num_residual_hiddens=32,
            embedding_dim=2,
            num_embeddings=3
        )
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_cherry_picked_loss((0, 1), (1, 0.25)), 1)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
