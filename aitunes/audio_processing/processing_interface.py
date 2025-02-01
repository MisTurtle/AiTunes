from os import path, makedirs
from typing import Literal, Union

import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


class AudioProcessingInterface:
    """
    A class allowing for an easy inline manipulation of audio files
    Usage Example:
        i = AudioProcessingInterface("path/to/file.wav")
        i.full_summary()  # Display features for the original file
        i.preprocess(fn1, fn2, fn3)  # Apply filters to the audio file
        i.full_summary()  # Display features for the preprocessed file
        i.save("path/to/new_file.wav")
        i.reset()  # Reset all changes and come back to the original file
    """

    @staticmethod
    def create_for(path: str, mode: Literal["file", "wave", "log_spec", "mel", "log_mel", "mfccs"], **kwargs):
        """
        Create audio from a .wav file
        """
        return AudioProcessingInterface(path, mode, **kwargs)


    def __init__(self, filepath: str, mode: Literal["file", "wave", "log_spec", "mel", "log_mel", "mfccs"], **kwargs):
        """
        :param path: Path to the .wav audio file
        :param sr: Sample rate for the WAV file, None means the audio isn't resampled
        """
        self._path = filepath
        self._sr = kwargs.get("sr", None)
        self._y = kwargs.get("data", None)
        self._label = kwargs.get("label", path.basename(self._path))
        kwargs.pop("data", None)
        kwargs.pop("label", None)

        match mode:
            case "file":
                self._y, self._sr = librosa.load(filepath, sr=self._sr)
            case "wave":
                pass
            case "log_spec":
                kwargs.pop("sr", None)
                self._y = librosa.griffinlim(librosa.db_to_amplitude(self._y), **kwargs)
            case "mel":
                self._y = librosa.feature.inverse.mel_to_audio(self._y, **kwargs)
            case "log_mel":
                self._y = librosa.feature.inverse.mel_to_audio(librosa.db_to_amplitude(self._y), **kwargs)
            case "mfccs":
                self._y = librosa.feature.inverse.mfcc_to_audio(self._y, **kwargs)
            
        if self._sr is None:
            raise ValueError("Error while creating an audio processing interface: Sample rate has to be defined")

    @property
    def duration(self) -> float:
        return self._y.shape[0] / self._sr
    
    def get_path(self) -> str:
        return self._path
    
    def get_label(self) -> str:
        """
        Label is used mostly in plots
        """
        return self._label
    
    def set_label(self, label: str) -> 'AudioProcessingInterface':
        self._label = label
        return self

    def get_data(self) -> tuple:
        return self._y, self._sr
    
    def copy(self, label: Union[str, None] = None) -> 'AudioProcessingInterface':
        return AudioProcessingInterface(self._path, mode='wave', data=self._y.copy(), sr=self._sr, label=label or (self.get_label + " (Copy)"))
    
    def preprocess(self, *fns) -> 'AudioProcessingInterface':
        for fn in fns:
            self._y = fn(self._y)
        return self
    
    def extract_window(self, duration: float, method: Literal["start", "end", "random", "bounded"] = "random", start: float = .0, label: Union[str, None] = None) -> 'AudioProcessingInterface':
        """
        Cuts the audio to a given duration
        :param duration: The target duration
        :param method: How to choose where to extract the window (Values are [start, end, random])
        """
        window_size = int(self._sr * duration)
        if window_size > self._y.shape[0]:
            return self
        
        match method:
            case "start":
                y = self._y[:window_size]
            case "end":
                y = self._y[-window_size:]
            case "bounded":
                if start + window_size > self._y.shape[0]:
                    y = self._y[-window_size:]
                else:
                    y = self._y[int(self._sr * start):int(self._sr * start) + window_size]
            case "random":
                start = np.random.randint(0, self._y.shape[0] - window_size)
                y = self._y[start:start+window_size]
        return AudioProcessingInterface(self.get_path(), mode="wave", data=y, sr=self._sr, label=label or (self.get_label() + " (Cropped)"))

    def log_spectrogram(self, **kwargs):
        return librosa.amplitude_to_db(np.abs(librosa.stft(self._y, **kwargs)) ** 2, ref=np.max)
    
    def mel_spectrogram(self, n_mels=128, **kwargs):
        return librosa.feature.melspectrogram(y=self._y, sr=self._sr, n_mels=n_mels, **kwargs)

    def log_mel_spectrogram(self, n_mels=128, **kwargs):
        return librosa.power_to_db(self.mel_spectrogram(n_mels, **kwargs), ref=np.max)

    def mfcc(self, n_mels=128, n_features=20, S=None):
        """
        :param n_features: Features resolution
        :param S: Precomputed spectogram (mel base)
        """
        if S is not None:
            return librosa.feature.mfcc(S=S, n_mels=n_mels, sr=self._sr, n_mfcc=n_features)
        return librosa.feature.mfcc(y=self._y, n_mels=n_mels, sr=self._sr, n_mfcc=n_features)    
    
    # Plotting functions
    def get_plot_for(
            self,
            features: list[Literal["wave", "log_spec", "log_mel", "mfcc", "empty"]],
            suptitle: Union[str, None] = "Summary",
            title: Union[None, str, list[str]] = None,
            wave_color: str = "b",
            wave_alpha: float = 1.,
            axes=None,
            fig=None,
            n_mels: int = 128,
            mfcc_features: int = 20
    ):
        if axes is None:
            fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(12, 10), constrained_layout=True, sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            for ax in axes:
                ax.xaxis.set_tick_params(labelbottom=True)
        else:
            assert fig is not None
            if not isinstance(axes, np.ndarray):
                axes = [axes]
        
        plt.suptitle(suptitle)
        if title is None or isinstance(title, str):
            title = [title] * len(features)
        
        for i in range(len(features)):
            match features[i]:
                case "wave":
                    self.draw_wave(axes[i], color=wave_color, alpha=wave_alpha)
                case "log_spec":
                    spec = self.log_spectrogram()
                    spec_img = librosa.display.specshow(spec, sr=self._sr, x_axis='time', y_axis='log', ax=axes[i])
                    plt.colorbar(spec_img, label="dB", ax=axes[i])
                    axes[i].set(title=title[i] or "Log Spectrogram")
                case "log_mel":
                    spec = self.log_mel_spectrogram(n_mels=n_mels)
                    spec_img = librosa.display.specshow(spec, sr=self._sr, x_axis='time', y_axis='mel', ax=axes[i])
                    plt.colorbar(spec_img, label="dB", ax=axes[i])
                    axes[i].set(title=title[i] or "Log Mel Spectrogram")
                case "mfcc":
                    mfcc = self.mfcc(n_features=mfcc_features)
                    mfcc_img = librosa.display.specshow(mfcc, sr=self._sr, x_axis='time', ax=axes[i])
                    plt.colorbar(mfcc_img, label="MFCC", ax=axes[i])
                    axes[i].set(title=title[i] or "MFCC Features")
        
        return fig, axes
    
    def draw_wave(self, ax, color: str = 'b', alpha: float = 1.0, label: Union[str, None] = None):
        librosa.display.waveshow(self._y, sr=self._sr, ax=ax, color=color, alpha=alpha, label=label or self.get_label())
        ax.set(title="Waveform", xlabel="Time", ylabel="Amplitude")
    
    def compare_waves(self, *others: tuple['AudioProcessingInterface'], include_self: bool = True, colors: list[str] = None, ax=None):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 2))

        alpha = 1. / (1 + len(others))
        colors = colors or ['b', 'r', 'g', 'y', 'orange', 'pink', 'brown', 'purple']

        if include_self:
            self.draw_wave(ax, alpha=alpha, color=colors[0])

        for i, interface in enumerate(others):
            interface.draw_wave(ax, color=colors[(i + 1) % len(colors)], alpha=alpha)
        
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.label_outer()
        ax.legend()

        return fig, ax

    def full_summary(self, compare_to: list['AudioProcessingInterface'] = [], n_mels=128, mfcc_features=20, save_to_dir=None, headless=False) -> 'AudioProcessingInterface':
        fig, axes = self.get_plot_for(
            features=["wave", "log_spec", "log_mel", "mfcc"],
            suptitle=f"Summary for {self.get_path()}",
            n_mels=n_mels,
            mfcc_features=mfcc_features,
            wave_alpha=1. / (1 + len(compare_to))
        )
        if len(compare_to) != 0:
            self.compare_waves(*compare_to, include_self=False, ax=axes[0])

        # enregistrer les spectogrammes
        if save_to_dir:
            if not os.path.exists(save_to_dir):
               os.makedirs(save_to_dir)
            save_path = os.path.join(save_to_dir, f"{path.basename(self._path)}_spectrogram.png")
            fig.savefig(save_path)

        # Afficher les graphiques
        if not headless:
            fig.show()
            fig.waitforbuttonpress()
        
        return self
    
    # Reconstruction tests

    def reconstruct_from_mel(self, n_mels=128) -> 'AudioProcessingInterface':
        """
        Simulates a reconstruction from a mel spectogram (Not in logarithmic scale)
        Might be expanded in the future to take in arbitrary spectrogram data depending on the NN output 
        """
        mel = self.mel_spectrogram(n_mels=n_mels)
        return AudioProcessingInterface(filepath=self.get_path(), mode="mel", data=mel, sr=self._sr)
    
    def reconstruct_from_mfcc(self, n_mels=128, mfcc_features=20) -> 'AudioProcessingInterface':
        """
        Simulates a reconstruction from mfcc features
        Might be expanded in the future to take in arbitrary mfcc data depending on the NN output
        """
        mfcc = self.mfcc(n_mels=n_mels, n_features=mfcc_features)
        return AudioProcessingInterface(filepath=self.get_path(), mode="mfccs", data=mfcc, sr=self._sr)
    
    # Play the audio
    
    def play(self, speed: float = 1.0):
        sd.play(self._y, self._sr * speed)

    # File operations

    def reset(self) -> 'AudioProcessingInterface':
        if not path.isfile(self._path):
            return self
        
        self._y, self._sr = librosa.load(self._path, sr=None)
        return self
    
    def save(self, outpath: Union[str, None]) -> 'AudioProcessingInterface':
        if outpath is None:
            outpath = self.get_path()
        if path.dirname(outpath) != "":
            makedirs(path.dirname(outpath), exist_ok=True)
        sf.write(outpath, self._y, self._sr)
        return self
