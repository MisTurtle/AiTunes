import random
import numpy as np
import torch
import torch.nn as nn
import aitunes.utils as utils

from collections import namedtuple
from os import path, walk
from typing import Callable, Union
from h5py import Dataset

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

from aitunes.audio_processing import PreprocessingCollection, AudioProcessingInterface
from aitunes.modules import AiTunesAutoencoderModule
from aitunes.utils import get_loading_char, append_to_dataset


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
    spec_preprocessing: Callable[[np.ndarray], np.ndarray],
    flush_every: int = 100  # How often to save spectrograms to the dataset file (reduce ram usage ._. we don't want 41 Gigs required lmao)
):
    if path.exists(training_output) and path.exists(evaluation_output):
        print(f"Precomputed spectrograms found at {training_output} and {evaluation_output}.")
        return
    assert path.exists(audio_folder)
    
    print(f"Precomputing spectrograms to {training_output} and {evaluation_output}")
    all_spectrograms, all_labels = [], []
    all_count, eval_count = 0, 0

    def flush():
        nonlocal all_count, eval_count
        # Have at least one spectrogram used as evaluation data per batch
        evaluation_data_size = max(1, int(len(all_spectrograms) * evaluation_proportions))
        evaluation_indices = np.random.choice(len(all_spectrograms), evaluation_data_size, replace=False)
        all_count += len(all_spectrograms)
        eval_count += evaluation_data_size
        # Split datasets
        training_data = np.delete(all_spectrograms, evaluation_indices, axis=0).astype(np.float32)
        training_labels = np.delete(all_labels, evaluation_indices, axis=0).astype("S")
        eval_data = np.take(all_spectrograms, evaluation_indices, axis=0).astype(np.float32)
        eval_labels = np.take(all_labels, evaluation_indices, axis=0).astype("S")
        # Append to file
        if len(training_data) > 0:
            append_to_dataset(training_output, { "spectrograms": training_data, "labels": training_labels })
        append_to_dataset(evaluation_output, { "spectrograms": eval_data, "labels": eval_labels })
        # Clear lists
        all_spectrograms.clear()
        all_labels.clear()

    for root, _, files in walk(audio_folder):
        for filename in files:  # Loop over audio files in the dataset
            if not utils.quiet:
                print(f"\r{get_loading_char()} Precomputing {filename}..." + " " * 10, end="")
            spectrograms, labels = precompute_spectrograms_for_audio_file(path.join(root, filename), features, audio_preprocessing, spec_preprocessing)
            all_spectrograms += spectrograms
            all_labels += labels

            # Save and clear memory every x samples
            if len(all_spectrograms) >= flush_every:
                flush()
    if len(all_spectrograms) > 0:
        flush()     
    print(f"\r{all_count} spectrograms precomputed (including {eval_count} for tests) and saved as HDF5 datasets...")

def precompute_spectrograms_for_audio_file(audio_path: str, features: AudioFeatures, audio_preprocessing: Callable[[np.ndarray], np.ndarray], spec_preprocessing: Callable[[np.ndarray], np.ndarray]):
    spectrograms, labels = [], []

    try:
        i = AudioProcessingInterface.create_for(audio_path, mode='file', sr=features.sample_rate)
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


def audio_model_interactive_evaluation(features: AudioFeatures, test_loader: Dataset, test_labels: Dataset, model: AiTunesAutoencoderModule, loss_criterion):
    """
    Inspired from https://matplotlib.org/stable/gallery/widgets/buttons.html
    
    Loops over validation data and show a comparison between the original audio and the reproduced one
    Controls for playing audio and switching tracks is also available
    
    Sorry if the function is hard to read ^_^ (collapse inner methods)
    """
    plt.switch_backend("TkAgg")

    # Create a new figure
    fig = plt.figure(figsize=(14, 8.5))
    gs = GridSpec(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
    axes = [
        fig.add_subplot(gs[0, :]),  # Wave form plot
        fig.add_subplot(gs[1, 0]),  # Original spectrogram
        fig.add_subplot(gs[1, 1]),  # Reconstructed spectrogram
        fig.add_subplot(gs[2, :])   # Latent space bar plot
    ]
    fig.subplots_adjust(bottom=0.09, top=0.925)

    # State tracker
    current_track: int = 0
    generated_from_scratch: bool = False
    original_spectrogram: Union[np.ndarray] = None
    original_label: Union[None, str] = None
    latent_sample: torch.Tensor = None
    loss: Union[None, torch.Tensor] = None
    og_interface: Union[None, AudioProcessingInterface] = None
    rec_interface: AudioProcessingInterface = None

    def update_state():
        """
        Takes the current track, generates a model prediction and creates audio interfaces for both the original and generated data
        If the state indicates to be generated from scratch, simply create an audio interface for the generated data
        """
        nonlocal model, original_spectrogram, original_label, latent_sample, og_interface, rec_interface, loss

        if generated_from_scratch:
            rec_spec = model.decode(latent_sample)
            original_spectrogram = original_label = og_interface = None
        else:
            original_spectrogram = test_loader[current_track]
            original_label = test_labels[current_track].decode('UTF-8')
            
            model_input = torch.tensor(original_spectrogram, dtype=torch.float32).unsqueeze(0)
            model_input = model_input.flatten(start_dim=1, end_dim=2) if model.flatten else model_input.unsqueeze(1)
            
            latent_sample, rec_spec, *args = model(model_input, training=False)
            loss, *_ = loss_criterion(model_input, rec_spec, *args)
            og_interface = reconstruct_audio(original_spectrogram, features, label=original_label)
        rec_interface = reconstruct_audio(rec_spec, features, label="Generated Track")

    # Button callbacks
    def next_track(incr: int):
        nonlocal current_track, generated_from_scratch, bplayog
        bplayog.set_active(True)
        generated_from_scratch = False
        current_track = (current_track + incr) % test_loader.shape[0]
        display_track()

    def play_original():
        if og_interface is not None:
            og_interface.play()

    def play_reconstructed():
        if rec_interface is not None:
            rec_interface.play()

    def generate_track():
        nonlocal generated_from_scratch, latent_sample, bplayog
        bplayog.set_active(False)
        generated_from_scratch = True
        latent_sample = model.sample(1)
        display_track()

    def display_track(colorbar: bool = False):
        """
        Display track data on the interface

        Args:
            colorbar (bool, optional): Add colorbar to spectrogram displays (only True on the first call, or they'll stack horizontally). Defaults to False.
        """
        update_state()
        for ax in axes:
            ax.clear()
        
        if generated_from_scratch:
            fig.suptitle("Randomly generated track")
            rec_interface.draw_wave(axes[0])
            rec_interface.get_plot_for(['log_mel'], title="Log Mel Spectrogram", axes=axes[2], fig=fig, colorbar=colorbar)
            axes[1].set_facecolor('lightgray')
            axes[1].text(0.5, 0.5, 'Nothing to Show', fontsize=16, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        else:
            fig.suptitle(f"Summary for reconstruction of {original_label} - Loss: {loss:.3f}")
            og_interface.compare_waves(rec_interface, ax=axes[0])
            og_interface.get_plot_for(['log_mel'], title="Original Log Mel Spectrogram", axes=axes[1], fig=fig, colorbar=colorbar)
            rec_interface.get_plot_for(['log_mel'], title="Reconstructed Log Mel Spectrogram", axes=axes[2], fig=fig, colorbar=colorbar)
        
        latent_flat_size = np.prod([*latent_sample.shape])
        axes[3].title.set_text("Latent Space Representation")        
        axes[3].bar(np.arange(0, latent_flat_size), latent_sample.flatten().cpu().tolist())
        if len(latent_sample.shape) > 2:
            for x in np.arange(0, latent_flat_size, latent_sample.shape[1]):
                axes[3].axvline(x - 0.5, color='r')  # Add a separator for each latent vector

    # Create buttons (from left to right)
    axgenerate  = fig.add_axes([0.10, 0.025, 0.08, 0.03])
    axreload    = fig.add_axes([0.19, 0.025, 0.08, 0.03])
    axplayog    = fig.add_axes([0.42, 0.025, 0.08, 0.03])
    axplayrec   = fig.add_axes([0.51, 0.025, 0.08, 0.03])
    axprev      = fig.add_axes([0.73, 0.025, 0.08, 0.03])
    axnext      = fig.add_axes([0.82, 0.025, 0.08, 0.03])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda _: next_track(1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda _: next_track(-1))
    bplayrec = Button(axplayrec, 'Play Generated')
    bplayrec.on_clicked(lambda _: play_reconstructed())
    bplayog = Button(axplayog, 'Play Original')
    bplayog.on_clicked(lambda _: play_original())
    breload = Button(axreload, 'Regenerate')
    breload.on_clicked(lambda _: display_track())  # This will regenerate a prediction based on the same input, eventually leading to a slightly different result due to normal distribution
    bgenerate = Button(axgenerate, 'Create Track')
    bgenerate.on_clicked(lambda _: generate_track())

    display_track(colorbar=True)
    plt.show()
