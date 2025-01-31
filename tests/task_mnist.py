import autoloader
import torch.nn as nn
import torch.optim as optim

from os import path
from torchsummary import summary
from aitunes.utils import simple_mse_kl_loss
from aitunes.autoencoders.task_cases import MnistDigitCompressionTaskCase, FLAG_PLOTTING
from aitunes.autoencoders.autoencoders_modules import SimpleAutoEncoder, VariationalAutoEncoder, CVAE


flags = FLAG_PLOTTING
epochs = 20


def ae(interactive: bool = True):
    model_path = path.join("assets", "Models", "ae_mnist.pth")
    model = SimpleAutoEncoder((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    loss, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)
    task = MnistDigitCompressionTaskCase(model, model_path, loss, optimizer, flatten=True, flags=flags)

    summary(model, (28 * 28, ))
    if not task.trained:
        task.train(epochs)
    task.evaluate()
    if interactive:
        task.display_embed_plot()
        task.interactive_evaluation()


def vae(interactive: bool = True):
    model_path = path.join("assets", "Models", "vae_mnist.pth")
    model = VariationalAutoEncoder((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
    task = MnistDigitCompressionTaskCase(model, model_path, loss, optimizer, flatten=True, flags=flags)

    summary(model, (28 * 28, ))
    if not task.trained:
        task.train(epochs)
    task.evaluate()
    if interactive:
        task.display_embed_plot()
        task.interactive_evaluation()


def cvae(interactive: bool = True):
    model_path = path.join("assets", "Models", "cvae_mnist.pth")
    model = CVAE(
        input_shape=[1, 28, 28],
        conv_filters=[32, 64, 128],
        conv_kernels=[ 3,  3,  3],
        conv_strides=[ 2,  2,  2],
        latent_space_dim=2
    )
    summary(model, (1, 28, 28))
    loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
    task = MnistDigitCompressionTaskCase(model, model_path, loss, optimizer, flatten=False, flags=flags)
    
    if not task.trained:
        task.train(epochs)
    task.evaluate()
    if interactive:
        task.display_embed_plot()
        task.interactive_evaluation()


if __name__ == "__main__":
    ae()
    vae()
    cvae()
