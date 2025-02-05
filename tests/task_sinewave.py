import autoloader
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Literal, Union
from os import makedirs, walk, path
from torchsummary import summary
from matplotlib import pyplot as plt

from aitunes.utils import get_loading_char, save_dataset, simple_mse_kl_loss
from aitunes.autoencoders.task_cases import SinewaveTaskCase, FLAG_PLOTTING, FLAG_NONE
from aitunes.autoencoders.autoencoders_modules import CVAE
from aitunes.audio_processing import AudioProcessingInterface, PreprocessingCollection
from aitunes.audio_generation.simple_audio_stream import generate_dataset_of_simple_instruments


# File system paths
dataset_path = path.join("assets", "Samples", "SINEWAVE")  # Dataset source path
audio_dataset_path = path.join("assets", "Samples", "SINEWAVE", "audio")  # Path to the dataset's audio files
precomputed_spectrograms_path = path.join(dataset_path, "spectrograms")  # Path to the dataset's precomputed mel spectograms
training_set_path = path.join(precomputed_spectrograms_path, "training_set.h5")  # Path to the precomputed training spectrograms dictionnary
evaluation_set_path = path.join(precomputed_spectrograms_path, "evaluation_set.h5")  # Path to the precomputed evaluation spectrograms dictionnary
comparison_path = path.join(dataset_path, "comparison")

# Features and settings
# -- Features to extract and recreate spectrograms
n_mels, n_fft, hop_length, sample_duration, sample_rate, sample_per_item = 256, 2048, 256, 5., 22050, 1
expected_spectrogram_size = n_mels, int(sample_duration * sample_rate // hop_length) + 1
flat_expected_spectrogram_size = expected_spectrogram_size[0] * expected_spectrogram_size[1]
# -- Data to use to denormalise and reconstruct audio from spectrograms
dB_bounds = -0.7, 0.7  # Spectrograms are mapped to this range once audio is recreated. Can be fine tuned later on
spect_bounds = -80, 0  # Bounds used to reconstruct audio from a normalized spectrogram. -80db to 0db

# H5 dataset resource endpoints
training_h5_file, evaluation_h5_file = None, None

# Save and history paths
release_root = path.join("assets", "Models", "sinewave")
history_root = path.join("history", "sinewave")

# Audio and model variables
flags = FLAG_PLOTTING
epochs = 100

def initial_setup():
    global training_h5_file, evaluation_h5_file
    
    if not path.exists(audio_dataset_path):
        generate_dataset_of_simple_instruments(audio_dataset_path, sample_rate=sample_rate, unit_duration=sample_duration)

    # Check if spectrograms have been precomputed
    if not path.exists(training_set_path) or not path.exists(evaluation_set_path):
        all_spectrograms, all_bounds = [], []
        for root, _, files in walk(audio_dataset_path):
            for filename in files:  # Loop over audio files in the dataset
                print(f"\r{get_loading_char()} Precomputing {filename}..." + " " * 10, end='')
                spectrograms, bounds = preprocess(path.join(root, filename))
                all_spectrograms += spectrograms
                all_bounds += bounds
        
        print("\rSpectrograms precomputed. Saving as HDF5 datasets...")
        
        all_data = list(zip(all_spectrograms, all_bounds))
        random.shuffle(all_data)
        all_spectrograms, all_bounds = zip(*all_data)

        # Save both datasets
        evaluation_data_size = len(all_spectrograms) // 10
        save_dataset(training_set_path, {"spectrograms": np.array(all_spectrograms[evaluation_data_size:]), "bounds": np.array(all_bounds[evaluation_data_size:])}, attrs={})
        save_dataset(evaluation_set_path, {"spectrograms": np.array(all_spectrograms[:evaluation_data_size]), "bounds": np.array(all_bounds[:evaluation_data_size])}, attrs={})
    else:
        print(f"Precomputed spectrograms found at {precomputed_spectrograms_path}")
    
    training_h5_file = h5py.File(training_set_path, mode="r+")
    evaluation_h5_file = h5py.File(evaluation_set_path, mode="r+")
    

def preprocess(audio_path: str):
    """
    Preprocess a file and return the precomputed spectrogram data as well as its original bounds
    :param path: Path to the .wav file
    """
    # Saving bounds seems actually kind of useless as every file in the dataset is normalized to [-80, 0] dB. But at least the system is in place for future expansion if necessary 
    spectrograms, bounds = [], []
    try:
        i = AudioProcessingInterface.create_for(audio_path, mode='file', sr=sample_rate)
        # Cut out the audio in small samples of appropriate duration
        samples_count = int( min( float(sample_per_item), i.duration / sample_duration ) )
        for sample_id in range(0, samples_count):
            # Cut out the window into the audio
            window = i.extract_window(sample_duration, method="bounded", start=sample_id * sample_duration)
            window.preprocess(preprocess_audio)
            
            # Compute its mel spectrogram
            spect = window.log_mel_spectrogram(n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            bounds.append([spect.min(), spect.max()])
            spect = preprocess_spectrogram(spect)

            if spect is None:
                raise Exception("Failed to preprocess spectrogram")
            
            # Test reconstruction before going forward with the whole dataset
            # spect = PreprocessingCollection.denormalise(spect, spect_bounds[0], spect_bounds[1])
            # w = AudioProcessingInterface.create_for("", "log_mel", data=spect, n_fft=n_fft, hop_length=hop_length, sr=sample_rate)
            # w.preprocess(lambda wave: PreprocessingCollection.denormalise(wave, dB_bounds[0], dB_bounds[1]))
            # window.play(blocking=True)
            # w.play(blocking=True)

            spectrograms.append(spect)
    except Exception as e:
        print(f"Skipped {audio_path} due to a raised error: ", e)
        return [], []
    return spectrograms, bounds

def preprocess_audio(y: np.ndarray):
    """
    Preprocessing applied directly to the audio data before precomputing the spectrogram
    :param y: The raw audio data
    """
    return y

def preprocess_spectrogram(y: np.ndarray):
    """
    Preprocessing applied to the spectrogram before being saved as a preprocessed numpy array
    :param y: The spectrogram data
    """
    return PreprocessingCollection.normalise(y, 0, 1)  # TODO : tests with log normalisation

def save_model_prediction(og: torch.Tensor, pred: torch.Tensor, embed: torch.Tensor, labels: list, _):
    """
    A middleware applied to verification data to save and compare audio between the original normalized spectrogram and the generated one
    TODO : Make this outdated by adding a "Save" button to the interactive evaluation
    """
    print("")  # Console formatting

    for k in range(og.shape[0]):
        spec_id = labels[0][k]
        print(f"\r{get_loading_char()} Reconstructing audio #{str(spec_id).zfill(3)}...", end='')

        normalized_spect = og[k]
        reconstruct_audio(normalized_spect, spec_id).save(path.join(comparison_path, f"{spec_id}_original.wav"))
        normalized_spect = pred[k]
        reconstruct_audio(normalized_spect, spec_id).save(path.join(comparison_path, f"{spec_id}_generated.wav"))


def reconstruct_audio(normalized_spectrogram: torch.Tensor, spec_id: int = -1, label: Union[str, None] = None) -> AudioProcessingInterface:
    bounds = spect_bounds
    normalized_spectrogram = normalized_spectrogram.reshape(expected_spectrogram_size)
    denormalized_spect = PreprocessingCollection.denormalise(normalized_spectrogram, bounds[0], bounds[1])

    if isinstance(denormalized_spect, torch.Tensor):
        denormalized_spect = denormalized_spect.cpu().numpy()

    i = AudioProcessingInterface.create_for("", mode="log_mel", data=denormalized_spect, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, label=label or f"Audio #{spec_id}")
    i.preprocess(lambda wave: PreprocessingCollection.denormalise(wave, dB_bounds[0], dB_bounds[1]))
    i.preprocess(preprocess_audio)
    return i


def cvae(evaluate: bool = True, interactive: bool = True):
    model_path = path.join(release_root, "cvae_sinewave.pth")
    history_path = path.join(history_root, "sinewave_v1")
    
    model = CVAE(
        input_shape=[1, *expected_spectrogram_size],
        conv_filters=[     32,      64,  128, 256, 512, 1024],
        conv_kernels=[      3,       3,    3,   3,   3,    3],
        conv_strides=[ (2, 1),  (2, 1),    2,   2,   2,    2],
        latent_space_dim=8
    )
    loss, optimizer = lambda *args: simple_mse_kl_loss(*args, beta=0.1), optim.Adam(model.parameters(), lr=0.001)
    
    task = SinewaveTaskCase(model, model_path, loss, optimizer, training_h5_file["spectrograms"], evaluation_h5_file["spectrograms"], reconstruct_audio, flatten=False, flags=flags)
    task.add_middleware(save_model_prediction)
    task.save_every(10, history_path)
    
    summary(model, (1, *expected_spectrogram_size))
    # if not task.trained:
    task.train(epochs)
    if evaluate:
        task.evaluate()
    if interactive:
        task.interactive_evaluation()


if __name__ == "__main__":
    initial_setup()
    
    cvae(evaluate=True)

    if training_h5_file is not None:
        training_h5_file.close()
    if evaluation_h5_file is not None:
        evaluation_h5_file.close()
