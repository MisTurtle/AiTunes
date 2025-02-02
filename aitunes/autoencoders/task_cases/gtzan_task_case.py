from typing import Union
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from aitunes.audio_processing.processing_interface import AudioProcessingInterface
from aitunes.utils import device
from aitunes.autoencoders.task_cases import AutoencoderTaskCase, FLAG_NONE

import h5py
import numpy as np
import torch

class GtzanDatasetTaskCase(AutoencoderTaskCase):
    """
    The GTZAN dataset has 100 30-second samples for a wapping 10 different genres, resulting in 1000 * 30 = 30'000 seconds of music to train on
    For this purpose, 95 samples will be split and used for the training process in each genre, while the other 5 will be used for model evaluation
    """
    def __init__(self, model, weights_path, loss, optimizer, training_data: h5py.File, evaluation_data: h5py.File, reconstruct_audio, flatten: bool = False, flags: int = FLAG_NONE):
        """
        :param reconstruct_audio: Reconstruct an AudioProcessingInterface from a normalized spectrogram
        """
        super().__init__("GTZAN", model, weights_path, loss, optimizer, flags)
        self._flatten = flatten
        self.train_loader = training_data
        self.test_loader = evaluation_data
        self.reconstruct_audio = reconstruct_audio

        # Interactive evaluation variables
        self._generated_sample: Union[None, torch.Tensor] = None  # Not None if the user is listening to brand new audio generated randomly
        self._current_track = 0
        self._axes, self._fig = None, None
        self._i_og: AudioProcessingInterface = None
        self._i_rec: AudioProcessingInterface = None  # i_og can be None if a new track was generated from a random latent sample
        self._latent_space_size = None  # Computed during interactive validation for now

    def next_batch(self, training): 
        dataset = self.train_loader
        if not training:
            dataset = self.test_loader
        
        complete = False
        batch_size, current_index = 50, 0
        while not complete:
            if current_index + batch_size >= dataset.shape[0]:
                spectrograms = dataset[current_index:]
                complete = True
            else:
                spectrograms = dataset[current_index:current_index + batch_size]
            spectrograms = torch.tensor(spectrograms, dtype=torch.float32)

            if self._flatten:
                spectrograms = spectrograms.flatten(start_dim=1, end_dim=2)
            else:
                spectrograms = spectrograms.unsqueeze(1)

            yield spectrograms, np.arange(current_index, current_index + spectrograms.shape[0])  # No real labels passed, but ids to the spectrograms.
            current_index += batch_size

    def interactive_evaluation(self):
        """
        Inspired from https://matplotlib.org/stable/gallery/widgets/buttons.html
        
        Loops over validation data and show a comparison between the original audio and the reproduced one
        Controls for playing audio and switching tracks is also available
        """
        with torch.no_grad():
            # Create the plot
            self._fig, self._axes = plt.subplots(ncols=2, figsize=(12, 5), width_ratios=(3, 1))
            self._fig.subplots_adjust(bottom=0.25, top=0.85)
            
            # Create buttons
            axgenerate = self._fig.add_axes([0.26, 0.05, 0.1, 0.075])
            axreload = self._fig.add_axes([0.37, 0.05, 0.1, 0.075])
            axplayrec = self._fig.add_axes([0.48, 0.05, 0.1, 0.075])
            axplayog = self._fig.add_axes([0.59, 0.05, 0.1, 0.075])
            axprev = self._fig.add_axes([0.7, 0.05, 0.1, 0.075])
            axnext = self._fig.add_axes([0.81, 0.05, 0.1, 0.075])

            bnext = Button(axnext, 'Next')
            bnext.on_clicked(self.next_track)
            bprev = Button(axprev, 'Previous')
            bprev.on_clicked(self.prev_track)
            bplayog = Button(axplayog, 'Play Original')
            bplayog.on_clicked(lambda _: self._i_og.play())
            bplayrec = Button(axplayrec, 'Play Generated')
            bplayrec.on_clicked(lambda _: self._i_rec.play())
            breload = Button(axreload, 'Regenerate')
            breload.on_clicked(self.display_track)  # This will regenerate a prediction based on the same input, eventually leading to a slightly different result due to normal distribution
            bgenerate = Button(axgenerate, 'Create Track')
            bgenerate.on_clicked(lambda _: self.generate_track())

            self.display_track()

            self._fig.show()
            while self._fig.waitforbuttonpress() == False:
                continue

    def generate_track(self):
        if self._latent_space_size is None:
            return  # Latent space size hasn't been computed yet, which shouldn't happen as it is computed right when we enter interactive mode
        self._generated_sample = torch.from_numpy(np.random.randn(self._latent_space_size).astype(np.float32)).to(device)
        self.display_track()

    def display_track(self, _=None):
        for ax in self._axes:
            ax.clear()
        
        if self._generated_sample is None:
            # Load the spectrogram and its prediction
            original_spectrogram = self.test_loader[self._current_track]
            model_input = torch.tensor(original_spectrogram, dtype=torch.float32)
            model_input = model_input.unsqueeze(0)

            # Reshape the input
            if self._flatten:
                model_input = model_input.flatten(start_dim=1, end_dim=2)
            else:
                model_input = model_input.unsqueeze(1)

            # Evaluate and predict the reconstructed spectrogram
            latent, reconstructed_spectrogram, *args = self.model(model_input)
            loss = self._loss_criterion(model_input, reconstructed_spectrogram, *args)
            self._fig.suptitle(f"Comparing Evaluation Track #{self._current_track}\nLoss: {loss}")

            # Fetch the audio processing interfaces from the middleware (not the cleanest but easier to implement)
            self._i_og = self.reconstruct_audio(original_spectrogram, self._current_track)
            self._i_rec = self.reconstruct_audio(reconstructed_spectrogram, self._current_track)
            self._i_og.compare_waves(self._i_rec, ax=self._axes[0])  # Draw the wave comparison
            self._latent_space_size = int(latent.shape[1])
            latent = latent[0]
        else:
            latent = self._generated_sample
            reconstructed_spectrogram = self.model._decoder(latent)
            self._i_og = self._i_rec = self.reconstruct_audio(reconstructed_spectrogram, -1)
            self._fig.suptitle("Randomly generated track")
            self._i_og.draw_wave(self._axes[0])

        # Plot the latent space
        self._axes[1].title.set_text("Latent Space State")
        self._axes[1].bar(np.arange(0, self._latent_space_size), latent.cpu().tolist())

    def next_track(self, _):
        self._generated_sample = None
        self._current_track = (self._current_track + 1) % self.test_loader.shape[0]
        self.display_track()
            
    def prev_track(self, _):
        self._generated_sample = None
        self._current_track = (self._current_track - 1) % self.test_loader.shape[0]
        self.display_track()
    

