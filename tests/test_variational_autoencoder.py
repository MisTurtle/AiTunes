import autoloader
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from aitunes.audio_processing import AudioProcessingInterface, PreprocessingCollection
from aitunes.autoencoders import VariationalAutoEncoder, CVAE
from aitunes.autoencoders.task_cases import *


def loss_criterion(prediction, target, mu, log_var):
    log_var = torch.clamp(log_var, max = 10.0)  # Prevent against KL_Divergence explosion
    reconstruction_loss = F.mse_loss(prediction, target)
    KL_Divergence = torch.mean(-.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))
    return reconstruction_loss + KL_Divergence
    

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
        conv_filters=[32, 64],
        conv_kernels=[3, 3],
        conv_strides=[1, 1],
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
    

def example4(model_weights: str = "assets/models/vae_gtzan.pth"):
    # Hop Length = How much data to skip ahead after each singular sampling point
    # N Mels = Y resolution for the mel-scaled spectrogram
    # Sample rate = How much recorded cycles per second (resampled by librosa if necessary)
    hop_length, n_mels, sr = 512, 64, 22050
    unit_audio_length = 1  # seconds
    expected_input_size = n_mels, unit_audio_length * sr // hop_length + 1
    flattened_input_size = expected_input_size[0] * expected_input_size[1]
    model = VariationalAutoEncoder([
        flattened_input_size,
        flattened_input_size // 2,
        flattened_input_size // 4,
        flattened_input_size // 8,
        flattened_input_size // 16
    ])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_NONE

    def path_to_trainable_audio(path: str):
        try:
            return AudioProcessingInterface.create_for(path, mode="file")\
                    .preprocess(lambda y: PreprocessingCollection.normalise(y, 0, 1))\
                    .log_mel_spectrogram(n_mels=n_mels)
        
        except Exception as e:
            print(f"/!\\ Skipped {path} due to error being thrown:", e)
            return np.zeros(expected_input_size)

    test_case = GtzanDatasetTestCase(model, model_weights, loss_criterion, optimizer, path_to_trainable_audio, flatten=True, flags=flags)
    if not test_case.trained:
        test_case.train(20)
    test_case.evaluate()


def example5(model_weights: str = "assets/models/cvae_gtzan.pth"):
    # Hop Length = How much data to skip ahead after each singular sampling point
    # N Mels = Y resolution for the mel-scaled spectrogram
    # Sample rate = How much recorded cycles per second (resampled by librosa if necessary)
    hop_length, n_mels, sr = 512, 128, 22050
    unit_audio_length = 1  # seconds
    expected_input_size = n_mels, unit_audio_length * sr // hop_length + 1
    
    model = CVAE(
        input_shape=[1, *expected_input_size],
        conv_filters=[64, 32, 16],
        conv_kernels=[3, 3, 3],
        conv_strides=[1, 1, 1],
        latent_space_dim=128
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_NONE

    def path_to_trainable_audio(path: str):
        try:
            spect = AudioProcessingInterface.create_for(path, mode="file", duration=unit_audio_length)\
                    .preprocess(lambda y: PreprocessingCollection.normalise(y, 0, 1))\
                    .log_mel_spectrogram(n_mels=n_mels)
            # test_i = AudioProcessingInterface.create_from_log_mel("assets/Samples/generated/gtzan/test.wav", spect, sr=sr)
            # test_i.summary("Actual original")
            # test_i.save(outpath=None)
            return spect
        
        except Exception as e:
            print(f"/!\\ Skipped {path} due to error being thrown:", e)
            return np.zeros(expected_input_size)

    test_case = GtzanDatasetTestCase(model, model_weights, loss_criterion, optimizer, path_to_trainable_audio, flatten=False, flags=flags)
    if not test_case.trained:
        test_case.train(10)
    test_case.evaluate()


def main(): 
    # example1()  # Vector3d VAE
    # example2()  # MNIST VAE
    # example3()  # MNIST CVAE
    # example4()  # GTZAN VAE
    example5()  # GTZAN CVAE

if __name__ == "__main__":
    main()
