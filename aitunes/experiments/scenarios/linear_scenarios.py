import torch.nn as nn
import torch.optim as optim

from os import path
from typing import Union
from aitunes.utils.loss_functions import *
from aitunes.experiments.cases import LinearExperiment
from aitunes.modules import SimpleAutoEncoder, VariationalAutoEncoder
from aitunes.experiments.scenarios._scenario_utils import ScenarioDescriptor, ScenarioContainer, scenario


class LinearAugmentationScenarios(ScenarioContainer):
   
    @property
    def identifier(self):
        return "5D VECTORS"

    @property
    def description(self):
        return "Vectors in five dimensions are created from three values, implying a dependency amongst those dimensions. This experiment aims to quickly try out simple architectures before moving on to more demanding tasks."
   
    @property
    def path_to_history_root(self):
        return path.join("history", "linear_augmentation")

    @property
    def path_to_release_root(self):
        return path.join("assets", "Models", "linear_augmentation")
 
    def instantiate(self, s: ScenarioDescriptor, model_path: Union[str, None]):
        model, loss, optimizer = s(self)
        return LinearExperiment(model, model_path or s.model_path, loss, optimizer)

    @scenario(name="AE", version="mse", description="A test to validate the SimpleAutoEncoder architecture")
    def ae(self):
        model = SimpleAutoEncoder((5, 4, 3))
        loss = create_mse_loss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
        
    @scenario(name="VAE", version="default", description="A test to validate the Variational AutoEncoder architecture. KL Loss weight: 0.001")
    def vae(self):
        model = VariationalAutoEncoder((5, 4, 3))
        loss = combine_losses(
            (create_mse_loss(reduction='mean'), 1),
            (create_kl_loss(reduce=True), 0.001)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
    