import torch.nn as nn
import torch.optim as optim

from os import path
from torchsummary import summary
from aitunes.utils import simple_mse_kl_loss
from aitunes.autoencoders.task_cases import LinearVectorAugmentationTaskCase, FLAG_PLOTTING
from aitunes.autoencoders.autoencoders_modules import SimpleAutoEncoder, VariationalAutoEncoder


flags = FLAG_PLOTTING
epochs = 200
release_root = path.join("assets", "Models", "5dvectors")
history_root = path.join("history", "5dvectors")


def ae(evaluation: bool = True, interactive: bool = True):
    model_path = path.join(release_root, "simple_ae.pth")
    history_path = path.join(history_root, "ae")

    model = SimpleAutoEncoder((5, 4, 3))
    loss, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)
    
    task = LinearVectorAugmentationTaskCase(model, model_path, loss, optimizer, flags)
    task.save_every(50, history_path)

    if not task.trained:
        task.train(epochs)
    if evaluation:
        task.evaluate()
        if interactive:
            task.interactive_evaluation()


def vae(evaluation: bool = True, interactive: bool = True):
    model_path = path.join(release_root, "simple_vae.pth")
    history_path = path.join(history_root, "vae")
    
    model = VariationalAutoEncoder((5, 4, 3))
    loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)

    task = LinearVectorAugmentationTaskCase(model, model_path, loss, optimizer, flags)
    task.save_every(50, history_path)

    if not task.trained:
        task.train(epochs)

    if evaluation:
        task.evaluate()
        if interactive:
            task.interactive_evaluation()


if __name__ == "__main__":
    # ae()
    vae()
