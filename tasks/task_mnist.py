import torch.nn as nn
import torch.optim as optim

from os import path
from torchsummary import summary
from aitunes.utils import simple_mse_kl_loss
from aitunes.autoencoders.task_cases import MnistDigitCompressionTaskCase, FLAG_PLOTTING
from aitunes.autoencoders.autoencoders_modules import SimpleAutoEncoder, VariationalAutoEncoder, CVAE


flags = FLAG_PLOTTING
epochs = 10
release_root = path.join("assets", "Models", "mnist")
history_root = path.join("history", "mnist")


def ae(evaluation: bool = True, interactive: bool = True):
    model_path = path.join(release_root, "ae_mnist.pth")
    history_path = path.join(history_root, "ae")
    
    model = SimpleAutoEncoder((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    loss, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)
    
    task = MnistDigitCompressionTaskCase(model, model_path, loss, optimizer, flatten=True, flags=flags)
    task.save_every(5, history_path)

    if not task.trained:
        task.train(epochs)

    if evaluation:
        task.evaluate()
        if interactive:
            task.display_embed_plot()
            task.interactive_evaluation()


def vae(evaluation: bool = True, interactive: bool = True):
    model_path = path.join(release_root, "vae_mnist.pth")
    history_path = path.join(history_root, "vae")
    
    model = VariationalAutoEncoder((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
    
    task = MnistDigitCompressionTaskCase(model, model_path, loss, optimizer, flatten=True, flags=flags)
    task.save_every(5, history_path)

    if not task.trained:
        task.train(epochs)
    if evaluation:
        task.evaluate()
        if interactive:
            task.display_embed_plot()
            task.interactive_evaluation()


def cvae(evaluation: bool = True, interactive: bool = True):
    model_path = path.join(release_root, "cvae_mnist.pth")
    history_path = path.join(history_root, "cvae")

    model = CVAE(
        input_shape=[1, 28, 28],
        conv_filters=[32, 64, 128],
        conv_kernels=[ 3,  3,  3],
        conv_strides=[ 2,  2,  2],
        latent_space_dim=2
    )
    loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
    
    task = MnistDigitCompressionTaskCase(model, model_path, loss, optimizer, flatten=False, flags=flags)
    task.save_every(5, history_path)

    if not task.trained:
        task.train(epochs)
    if evaluation:
        task.evaluate()
        if interactive:
            task.display_embed_plot()
            task.interactive_evaluation()


if __name__ == "__main__":
    # ae()
    # vae()
    cvae()
