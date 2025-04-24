
import autoloader

import torch.optim as optim
from aitunes.modules import VanillaAutoEncoder
from aitunes.experiments.cases import LinearExperiment
from aitunes.utils.loss_functions import create_mse_loss

model = VanillaAutoEncoder(input_shape=5, hidden_layer_dimensions=[8, 7, 6], latent_dimension=5)
loss = create_mse_loss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001)

experiment = LinearExperiment(model, weights_path='model.pth', loss=loss, optimizer=optimizer)
experiment.save_every(epoch_count=100, root_folder='.')

experiment.train(epochs=300)
experiment.evaluate()
experiment.interactive_evaluation()
