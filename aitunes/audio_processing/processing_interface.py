from os import path, makedirs
from typing import Literal, Union

import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


class AudioProcessingInterface:
    """
    A class allowing for an easy inline manipulation of audio files
    Usage Example:
        i = AudioProcessingInterface("path/to/file.wav")
        i.summary()  # Display features for the original file
        i.preprocess(fn1, fn2, fn3)  # Apply filters to the audio file
        i.summary()  # Display features for the preprocessed file
        i.save("path/to/new_file.wav")
        i.reset()  # Reset all changes and come back to the original file
    """

    @staticmethod
    def create_for(path: str, mode: Literal["file", "wave", "log_spec", "mel", "log_mel", "mfccs"], **kwargs):
        """
        Create audio from a .wav file
        """
        return AudioProcessingInterface(path, mode, **kwargs)


    def __init__(self, path: str, mode: Literal["file", "wave", "log_spec", "mel", "log_mel", "mfccs"], **kwargs):
        """
        :param path: Path to the .wav audio file
        :param sr: Sample rate for the WAV file, None means the audio isn't resampled
        """
        self._path = path
        self._sr = kwargs.get("sr", None)
        self._y = kwargs.get("data", None)
        kwargs.pop("data", None)

        match mode:
            case "file":
                self._y, self._sr = librosa.load(path, sr=self._sr)
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

    def get_data(self) -> tuple:
        return self._y, self._sr
    
    def extract_window(self, duration: float, method: Literal["start", "end", "random", "bounded"] = "random", start: float = .0) -> 'AudioProcessingInterface':
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
        return AudioProcessingInterface(self.get_path(), mode="wave", data=y, sr=self._sr)

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
    

    def summary(self, qualifier: str="", n_mels=128, mfcc_features=20, save_to_dir=None, headless=False) -> 'AudioProcessingInterface':
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), constrained_layout=True, sharex=True)
        for a in ax:
            a.xaxis.set_tick_params(labelbottom=True)
            a.set_xlim([0, self.duration])

        # Waveform
        librosa.display.waveshow(self._y, sr=self._sr, ax=ax[0])
        ax[0].set(title="Waveform", xlabel="Time", ylabel="Amplitude")
        
        # Log Spectrogram
        spec = self.log_spectrogram()
        spec_img = librosa.display.specshow(spec, sr=self._sr, x_axis='time', y_axis='log', ax=ax[1])
        fig.colorbar(spec_img, label="dB", ax=ax[1])
        ax[1].set(title="Log Spectrogram")

        # Mel Spectrogram
        spec = self.log_mel_spectrogram(n_mels=n_mels)
        spec_img = librosa.display.specshow(spec, sr=self._sr, x_axis='time', y_axis='mel', ax=ax[2])
        fig.colorbar(spec_img, label="dB", ax=ax[2])
        ax[2].set(title="Log Mel Spectrogram")

        # MFCC Features
        mfcc = self.mfcc(n_features=mfcc_features)
        mfcc_img = librosa.display.specshow(mfcc, sr=self._sr, x_axis='time', ax=ax[3])
        fig.colorbar(mfcc_img, label="MFCC", ax=ax[3])
        ax[3].set(title="MFCC Features")

        plt.suptitle(qualifier + " > " + path.basename(self._path))

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

    def reconstruct_from_mel(self, n_mels=128) -> 'AudioProcessingInterface':
        """
        Simulates a reconstruction from a mel spectogram (Not in logarithmic scale)
        Might be expanded in the future to take in arbitrary spectrogram data depending on the NN output 
        """
        mel = self.mel_spectrogram(n_mels=n_mels)
        return AudioProcessingInterface(path=self.get_path(), mode="mel", data=mel, sr=self._sr)
    
    def reconstruct_from_mfcc(self, n_mels=128, mfcc_features=20) -> 'AudioProcessingInterface':
        """
        Simulates a reconstruction from mfcc features
        Might be expanded in the future to take in arbitrary mfcc data depending on the NN output
        """
        mfcc = self.mfcc(n_mels=n_mels, n_features=mfcc_features)
        return AudioProcessingInterface(path=self.get_path(), mode="mfccs", data=mfcc, sr=self._sr)

    def preprocess(self, *fns) -> 'AudioProcessingInterface':
        for fn in fns:
            self._y = fn(self._y)
        return self
    
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
        