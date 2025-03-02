import torch.optim as optim

from os import path
from typing import Union
from aitunes.modules import *
from aitunes.utils.loss_functions import *
from aitunes.experiments.cases import LinearExperiment
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario


class LinearAugmentationScenarios(ScenarioContainer):
   
    @property
    def identifier(self):
        return "5D VECTORS"

    @property
    def description(self):
        return "Vectors in five dimensions are created from three values, implying a dependency amongst those dimensions. This experiment aims to quickly try out simple architectures before moving on to more demanding tasks."
   
    @property
    def path_to_history_root(self):
        return path.join("history", "vectors_5d")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "linear_augmentation")
 
    def instantiate(self, s, model_path):
        model, loss, optimizer = s(self)
        return LinearExperiment(model, model_path or s.model_path, loss, optimizer)

    @scenario(name="AE", version="hello_world", description="A test to validate the SimpleAutoEncoder architecture")
    def ae_hello_world(self):
        model = VanillaAutoEncoder(5, [4], 3)
        loss = create_mse_loss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="AE", version="deep_world", description="A test to validate the SimpleAutoEncoder architecture with a deeper network")
    def ae_deep_world(self):
        model = VanillaAutoEncoder(5, [8, 6, 6, 8], 3)
        loss = create_mse_loss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
        
    @scenario(name="VAE", version="hello_kl", description="A test to validate the Variational AutoEncoder architecture. KL Loss weight: 0.01")
    def vae_hello_kl(self):
        model = VariationalAutoEncoder(5, [8, 6, 6, 8], 3)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss(reduce=True), 0.01)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    
    @scenario(name="VAE", version="small_kl", description="A test to validate the Variational AutoEncoder architecture giving less importance to the KL Divergence Loss. KL Loss weight: 0.001")
    def vae_small_kl(self):
        model = VariationalAutoEncoder(5, [8, 6, 6, 8], 3)
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss(reduce=True), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    