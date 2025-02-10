import torch.nn as nn
import torch.optim as optim

from os import path
from typing import Union
from aitunes.utils.loss_functions import simple_mse_kl_loss, mse_loss
from aitunes.experiments.cases import LinearExperiment
from aitunes.modules import SimpleAutoEncoder, VariationalAutoEncoder
from aitunes.experiments.scenarios._scenario_utils import ScenarioDescriptor, ScenarioContainer, scenario


class LinearAugmentationScenarios(ScenarioContainer):
    
    @staticmethod
    def get_path_to_history_root():
        return path.join("history", "linear_augmentation")

    @staticmethod
    def get_path_to_release_root():
        return path.join("assets", "Models", "linear_augmentation")
    
    @staticmethod
    def get_identifier():
        return "VectorAugmentation"
    
    @staticmethod
    def get_description():
        return "Vectors in five dimensions are created from three values, implying a dependency amongst those dimensions. This experiment aims to quickly try out simple architectures before moving on to more demanding tasks."
    
    def instantiate(self, s: ScenarioDescriptor, model_path: Union[str, None]):
        model, loss, optimizer = s(self)
        return LinearExperiment(model, model_path or s.model_path, loss, optimizer)

    @scenario(name="Simple AE", version="1.0", description="A test to validate the SimpleAutoEncoder architecture")
    def ae(self):
        model = SimpleAutoEncoder((5, 3))
        loss, optimizer = mse_loss, optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
        
    @scenario(name="Simple VAE", version="1.0", description="A test to validate the Variational AutoEncoder architecture")
    def vae(self):
        model = VariationalAutoEncoder((5, 3))
        loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
        return model, loss, optimizer
