import autoloader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from aitunes.audio_processing.processing_interface import AudioProcessingInterface
from aitunes.autoencoders import VariationalAutoEncoder, CVAE
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
    test_case = MnistDigitCompressionTestCase(model, model_weights, loss_criterion, optimizer, flags, flatten=True, flags=flags)
    
    if not test_case.trained:
        test_case.train(10)
    test_case.evaluate()
    test_case.display_embed_plot()
    test_case.interactive_evaluation()


def example3(model_weights: str = "assets/models/cvae_mnist.pth"):
    model = CVAE(
        input_shape=[1, 28, 28],
        conv_filters=[2, 2, 2],
        conv_kernels=[3, 3, 3],
        conv_strides=[1, 1, 1],
        latent_space_dim=2
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_PLOTTING
    test_case = MnistDigitCompressionTestCase(model, model_weights, loss_criterion, optimizer, flatten=False, flags=flags)
    
    if not test_case.trained:
        test_case.train(10)
    test_case.evaluate()
    test_case.display_embed_plot()
    test_case.interactive_evaluation()
    

def example4(model_weights: str = "assets/models/cvae_gtzan.pth"):
    # Hop Length = How much data to skip ahead after each singular sampling point
    # N Mels = Y resolution for the mel-scaled spectrogram
    # Sample rate = How much recorded cycles per second (resampled by librosa if necessary)
    hop_length, n_mels, sr = 512, 64, 22050 
    unit_audio_length = 5  # seconds

    # Window size to cut out only a small part into the audio
    complete_window_size = sr * unit_audio_length
    actual_window_size = complete_window_size // hop_length

    model = CVAE((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_PLOTTING

    def path_to_trainable_audio(path: str):
        interface = AudioProcessingInterface.create_for(path)
        interface.preprocess(

        )
        spectrogram = interface.log_mel_spectrogram(n_mels=n_mels)
        print(interface._y.shape, spectrogram.shape, interface._sr)
        return spectrogram[:actual_window_size, :]  # TODO : Change this to a random window in the spectrogram
    
    test_case = GtzanDatasetTestCase(model, model_weights, loss_criterion, optimizer, path_to_trainable_audio, flags)
    if not test_case.trained:
        test_case.train(10)
    test_case.evaluate()


def main(): 
    # example1()
    # example2()
    example3()

if __name__ == "__main__":
    main()
