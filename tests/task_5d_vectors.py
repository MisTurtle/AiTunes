import autoloader
import torch.nn as nn
import torch.optim as optim

from os import path
from torchsummary import summary
from aitunes.utils import simple_mse_kl_loss
from aitunes.autoencoders.task_cases import LinearVectorAugmentationTaskCase, FLAG_PLOTTING
from aitunes.autoencoders.autoencoders_modules import SimpleAutoEncoder, VariationalAutoEncoder


flags = FLAG_PLOTTING
epochs = 200


def ae(interactive: bool = True):
    model_path = path.join("assets", "Models", "ae_5d_vectors.pth")
    model = SimpleAutoEncoder((5, 4, 3))
    loss, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)
    task = LinearVectorAugmentationTaskCase(model, model_path, loss, optimizer, flags)

    summary(model, (5, ))
    if not task.trained:
        task.train(epochs)
    task.evaluate()
    if interactive:
        task.interactive_evaluation()


def vae(interactive: bool = True):
    model_path = path.join("assets", "Models", "vae_5d_vectors.pth")
    model = VariationalAutoEncoder((5, 4, 3))
    loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
    task = LinearVectorAugmentationTaskCase(model, model_path, loss, optimizer, flags)

    summary(model, (5, ))
    if not task.trained:
        task.train(epochs)
    task.evaluate()
    if interactive:
        task.interactive_evaluation()


if __name__ == "__main__":
    ae()
    vae()
