import autoloader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from aitunes.autoencoders import VariationalAutoEncoder
from aitunes.autoencoders.task_cases import *


def loss_criterion(prediction, target, mu, log_var):
    reconstruction_loss = F.mse_loss(prediction, target)
    KL_Divergence = torch.mean(-.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))
    return 100 * reconstruction_loss + KL_Divergence
    

def example1(model_weights: str = "assets/models/vae_5d_vectors.pth"):
    model = VariationalAutoEncoder((5, 4, 3))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_PLOTTING
    test_case = LinearVectorAugmentationTestCase(model, model_weights, loss_criterion, optimizer, flags)
    
    if not test_case.trained:
        test_case.train(300)
    test_case.evaluate()
    test_case.interactive_evaluation()


def example2(model_weights: str = "assets/models/vae_mnist.pth"):
    model = VariationalAutoEncoder((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_PLOTTING
    test_case = MnistDigitCompressionTestCase(model, model_weights, loss_criterion, optimizer, flags)
    
    if not test_case.trained:
        test_case.train(10)
    test_case.evaluate()
    test_case.display_embed_plot()
    test_case.interactive_evaluation()


def main(): 
    # example1()
    example2()

if __name__ == "__main__":
    main()
