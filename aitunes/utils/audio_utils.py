from collections import namedtuple
from os import path, walk
import random
from typing import Callable, Union

import numpy as np
import torch

from aitunes.audio_processing import PreprocessingCollection, AudioProcessingInterface
from aitunes.utils import get_loading_char, save_dataset


class AudioFeatures(namedtuple("AudioFeatures", field_names=["n_mels", "n_mfcc", "n_fft", "hop_length", "sample_rate", "duration", "spec_bounds", "dB_bounds"])):
    # This only supports log_mel spectrograms for now. Maybe add setter properties for the type later on to recompute these: 
    @property
    def spectrogram_size(self):
        return self.n_mels, int(self.duration * self.sample_rate // self.hop_length) + 1
    
    @property
    def flat_spectrogram_size(self):
        s = self.spectrogram_size
        return s[0] * s[1]

def HighResolutionAudioFeatures(duration: float, spec_bounds: tuple[float, float] = (-80, 0), dB_bounds: tuple[float, float] = (-0.7, 0.7)):
    return AudioFeatures(n_mels=256, n_mfcc=64, n_fft=2048, hop_length=256, sample_rate=22050, duration=duration, spec_bounds=spec_bounds, dB_bounds=dB_bounds)

def LowResolutionAudioFeatures(duration: float, spec_bounds: tuple[float, float] = (-80, 0), dB_bounds: tuple[float, float] = (-0.7, 0.7)):
    return AudioFeatures(n_mels=64, n_mfcc=32, n_fft=1024, hop_length=512, sample_rate=22050, duration=duration, spec_bounds=spec_bounds, dB_bounds=dB_bounds)

def reconstruct_audio(n_spectrogram: torch.Tensor, features: AudioFeatures, preprocessing_fn: Union[None, Callable[[np.ndarray], np.ndarray]] = None, label: Union[str, None] = None) -> AudioProcessingInterface:
    """
    Reconstruct an audio interface from a normalized log-mel spectrogram
    :param n_spectrogram: A normalized spectrogram (can be linear or reshaped, it doesn't really matter)
    :param spec_bounds: Spectrogram bounds to denormalize to
    :param dB_bounds: Decibel bounds to map the audio to
    :param label: Label to be given to the audio interface
    """
    n_spectrogram = n_spectrogram.reshape(features.spectrogram_size)
    d_spectrogram = PreprocessingCollection.denormalise(n_spectrogram, *features.spec_bounds)

    if isinstance(d_spectrogram, torch.Tensor):
        d_spectrogram = d_spectrogram.cpu().numpy()
    
    i = AudioProcessingInterface.create_for("", mode="log_mel", data=d_spectrogram, sr=features.sample_rate, n_fft=features.n_fft, hop_length=features.hop_length, label=label or "Audio Track")
    i.preprocess(lambda wave: PreprocessingCollection.denormalise(wave, *features.dB_bounds))
    if preprocessing_fn is not None:
        i.preprocess(preprocessing_fn)
    return i

def precompute_spectrograms_for_audio_folder(
    audio_folder: str,
    training_output: str, evaluation_output: str, evaluation_proportions: float,
    features: AudioFeatures,
    audio_preprocessing: Callable[[np.ndarray], np.ndarray],
    spec_preprocessing: Callable[[np.ndarray], np.ndarray]
):
    assert path.exists(audio_folder)
    if path.exists(training_output) and path.exists(evaluation_output):
        print(f"Precomputed spectrograms found at {training_output} and {evaluation_output}.")
        return
    
    print(f"Precomputing spectrograms to {training_output} and {evaluation_output}")
    all_spectrograms, all_labels = [], []
    for root, _, files in walk(audio_folder):
        for filename in files:  # Loop over audio files in the dataset
            print(f"\r{get_loading_char()} Precomputing {filename}..." + " " * 10, end="")
            spectrograms, labels = precompute_spectrograms_for_audio_file(path.join(audio_folder, filename), features, audio_preprocessing, spec_preprocessing)
            all_spectrograms += spectrograms
            all_labels += labels

    print(f"\r{len(all_spectrograms)} spectrograms precomputed. Now saving as HDF5 datasets...")

    # Shuffle the spectrograms by keeping their associated label
    all_data = list(zip(all_spectrograms, all_labels))
    random.shuffle(all_data)
    all_spectrograms, all_labels = zip(*all_data)

    # Save training and evaluation datasets
    evaluation_data_size = int(len(all_spectrograms) * evaluation_proportions)
    save_dataset(training_output,   { "spectrograms": np.array(all_spectrograms[evaluation_data_size:]), "labels": np.array(all_labels[evaluation_data_size:], dtype="S") })
    save_dataset(evaluation_output, { "spectrograms": np.array(all_spectrograms[:evaluation_data_size]), "labels": np.array(all_labels[:evaluation_data_size], dtype="S") })

def precompute_spectrograms_for_audio_file(audio_path: str, features: AudioFeatures, audio_preprocessing: Callable[[np.ndarray], np.ndarray], spec_preprocessing: Callable[[np.ndarray], np.ndarray]):
    spectrograms, labels = [], []

    try:
        i = AudioProcessingInterface.create_for(audio_path, mode='file', sample_rate=features.sample_rate)
        samples_n = int(i.duration // features.duration)

        # Cut out the audio in small samples of appropriate duration
        for n in range(0, samples_n):
            # Extract the window
            window = i.extract_window(features.duration, method="bounded", start=n * features.duration)
            window.preprocess(audio_preprocessing)
            
            # Compute its mel spectrogram
            spect = window.log_mel_spectrogram(n_mels=features.n_mels, n_fft=features.n_fft, hop_length=features.hop_length)
            spect = spec_preprocessing(spect)

            if spect is None:
                raise Exception("Failed to preprocess spectrogram")
            
            # Test reconstruction before going forward with the whole dataset
            # test_spect = PreprocessingCollection.denormalise(spect, *features.spec_bounds)
            # test_w = AudioProcessingInterface.create_for("", mode="log_mel", data=test_spect, n_fft=features.n_fft, hop_length=features.hop_length, sr=features.sample_rate)
            # test_w.preprocess(lambda wave: PreprocessingCollection.denormalise(wave, *features.dB_bounds))
            # window.play(blocking=True)
            # test_w.play(blocking=True)

            spectrograms.append(spect)
            labels.append(path.basename(audio_path))
    except Exception as e:
        print(f"Skipped {audio_path} due to a raised error: ", str(e))
    
    return spectrograms, labels
