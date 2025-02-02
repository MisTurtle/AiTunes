
import autoloader
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from os import listdir, makedirs, walk, path
from torchsummary import summary
from matplotlib import pyplot as plt

from aitunes.utils import download_and_extract, get_loading_char, save_dataset, simple_mse_kl_loss
from aitunes.autoencoders.task_cases import GtzanDatasetTaskCase, FLAG_PLOTTING, FLAG_NONE
from aitunes.autoencoders.autoencoders_modules import VariationalAutoEncoder, CVAE
from aitunes.audio_processing import AudioProcessingInterface, PreprocessingCollection


# File system paths
dataset_path = path.join("assets", "Samples", "GTZAN")
audio_dataset_path = path.join(dataset_path, "genres_original")
precomputed_spectrograms_path = path.join(dataset_path, "spectrograms")
training_set_path, evaluation_set_path = path.join(precomputed_spectrograms_path, "training_set.h5"), path.join(precomputed_spectrograms_path, "evaluation_set.h5")
comparison_path = path.join("assets", "Samples", "generated", "gtzan")
makedirs(comparison_path, exist_ok=True)

# Audio and model variables
flags = FLAG_NONE
epochs = 1000
n_fft, hop_length, sample_duration, sample_rate, sample_per_item = 1024, 512, 1., 22050, 2
expected_spectrogram_size = int(n_fft // 2) + 1, int(sample_duration * sample_rate // hop_length) + 1
flat_expected_spectrogram_size = expected_spectrogram_size[0] * expected_spectrogram_size[1]
dB_bounds = -0.5, 0.5  # Spectrograms are mapped to this range once audio is recreated. Can be fine tuned later on
spec_bounds = -80, 0  # Bounds used to reconstruct audio from a normalized spectrogram. -80db to 0db

# H5 file instances
training_h5_file, evaluation_h5_file = None, None

def initial_setup():
    global training_h5_file, evaluation_h5_file
    # Ensure the dataset is downloaded at the expected location
    download_and_extract(
        url="https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification",
        target_path=dataset_path,
        zip_path=path.join(dataset_path, "..", "GTZAN_Dataset.zip"),
        final_size=1241.20,  # Dataset final size is 1.241 GB
        standalone_zipped_dir="Data",  # A Data folder is at the root of the downloaded zip file
        clean=True  # Cleanup the downloaded zip file once finished
    )

    # Check if spectrograms have been precomputed
    if not path.exists(training_set_path) or not path.exists(evaluation_set_path):
        evaluation_data_size = 50 * sample_per_item  # (1000 items in the dataset, 100 used for evaluation)
        makedirs(precomputed_spectrograms_path, exist_ok=True)
        all_spectrograms, all_bounds = [], []
        for root, _, files in walk(audio_dataset_path):
            for filename in files:  # Loop over audio files in the dataset
                print(f"\r{get_loading_char()} Precomputing {filename}..." + " " * 10, end='')
                spectrograms, bounds = preprocess(path.join(root, filename))
                all_spectrograms += spectrograms
                all_bounds += bounds
        
        print(all_bounds)        
        print("\rSpectrograms precomputed. Saving as HDF5 datasets...")
        # attrs = {"n_fft": n_fft, "hop_length": hop_length, "sample_duration": sample_duration, "sample_rate": sample_rate}  # Might be useful to store those? Meh maybe not
        
        all_data = list(zip(all_spectrograms, all_bounds))
        random.shuffle(all_data)
        all_spectrograms, all_bounds = zip(*all_data)

        save_dataset(training_set_path, {
            "spectrograms": np.array(all_spectrograms[evaluation_data_size:]),
            "bounds": np.array(all_bounds[evaluation_data_size:])
        }, attrs={})
        save_dataset(evaluation_set_path, {
            "spectrograms": np.array(all_spectrograms[:evaluation_data_size]),
            "bounds": np.array(all_bounds[:evaluation_data_size])
        }, attrs={})
    else:
        print(f"Precomputed spectrograms found at {precomputed_spectrograms_path}")
    
    training_h5_file, evaluation_h5_file = h5py.File(training_set_path, "r+"), h5py.File(evaluation_set_path, "r+")


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
        for start in range(0, int(min(float(sample_per_item), i.duration / sample_duration))):
            # Cut out the window into the audio
            window = i.extract_window(sample_duration, method="bounded", start=start * sample_duration)
            window.preprocess(preprocess_audio)
            # Compute its spectrogram
            spect = window.log_spectrogram(n_fft=n_fft, hop_length=hop_length) 
            bounds.append([spect.min(), spect.max()])
            spect = preprocess_spectrogram(spect)
            if spect is None:
                raise Exception("Failed to preprocess spectrogram")
            # Save to file
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
    return PreprocessingCollection.normalise(y, 0, 1)

def save_model_prediction(og: torch.Tensor, pred: torch.Tensor, embed: torch.Tensor, labels: list, _):
    """
    A middleware applied to verification data to save and compare audio between the original normalized spectrogram and the generated one
    """
    print("")  # Console formatting

    for k in range(og.shape[0]):
        spec_id = labels[0][k]
        print(f"\r{get_loading_char()} Reconstructing audio #{str(spec_id).zfill(3)}...", end='')

        normalized_spect = og[k]
        reconstruct_audio(normalized_spect, spec_id).save(path.join(comparison_path, f"original_{spec_id}.wav"))
        normalized_spect = pred[k]
        reconstruct_audio(normalized_spect, spec_id).save(path.join(comparison_path, f"generated_{spec_id}.wav"))


def reconstruct_audio(normalized_spectrogram: torch.Tensor, spec_id: int = -1) -> AudioProcessingInterface:
    if spec_id < 0:
        bounds = spec_bounds
    else:
        bounds = evaluation_h5_file["bounds"][spec_id]

    normalized_spectrogram = normalized_spectrogram.reshape(expected_spectrogram_size)
    denormalized_spect = PreprocessingCollection.denormalise(normalized_spectrogram, bounds[0], bounds[1])

    if isinstance(denormalized_spect, torch.Tensor):
        denormalized_spect = denormalized_spect.cpu().numpy()

    i = AudioProcessingInterface.create_for("", mode="log_spec", data=denormalized_spect, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    i.preprocess(lambda wave: PreprocessingCollection.denormalise(wave, dB_bounds[0], dB_bounds[1]))
    return i


def vae(evaluate: bool = True, interactive: bool = True):
    model_path = path.join("assets", "Models", "vae_gtzan.pth")
    model = VariationalAutoEncoder((
        flat_expected_spectrogram_size,
        flat_expected_spectrogram_size // 8,
        flat_expected_spectrogram_size // 16,
        flat_expected_spectrogram_size // 32,
        16
    ))
    loss, optimizer = lambda *args: simple_mse_kl_loss(*args, reconstruction_weight=100000), optim.Adam(model.parameters(), lr=0.001)
    task = GtzanDatasetTaskCase(model, model_path, loss, optimizer, training_h5_file["spectrograms"], evaluation_h5_file["spectrograms"], reconstruct_audio, flatten=True, flags=flags)
    task.add_middleware(save_model_prediction)

    summary(model, (flat_expected_spectrogram_size, ))
    if not task.trained:
        task.train(epochs)
    if evaluate:
        task.evaluate()
    if interactive:
        task.interactive_evaluation()


def cvae(interactive: bool = True):
    pass
    # TODO.
    # model_path = path.join("assets", "Models", "cvae_mnist.pth")
    # model = CVAE(
    #     input_shape=[1, 28, 28],
    #     conv_filters=[32, 64, 128],
    #     conv_kernels=[ 3,  3,  3],
    #     conv_strides=[ 2,  2,  2],
    #     latent_space_dim=2
    # )
    # summary(model, (1, 28, 28))
    # loss, optimizer = simple_mse_kl_loss, optim.Adam(model.parameters(), lr=0.001)
    # task = GtzanDatasetTaskCase(model, model_path, loss, optimizer, training_h5_file["spectrograms"], evaluation_h5_file["spectrograms"], reconstruct_audio, flatten=False, flags=flags)
    
    # if not task.trained:
    #     task.train(epochs)
    # task.evaluate()


if __name__ == "__main__":
    initial_setup()
    vae(evaluate=False)
    cvae()
    if training_h5_file is not None:
        training_h5_file.close()
    if evaluation_h5_file is not None:
        evaluation_h5_file.close()
