from os import path, makedirs
from typing import Union
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os


class AudioProcessingInterface:
    """
    A class allowing for an easy inline manipulation of audio files
    Usage Example:
        i = AudioProcessingInterface("path/to/file.wav")
        i.display()  # Display features for the original file
        i.preprocess(fn1, fn2, fn3)  # Apply filters to the audio file
        i.display()  # Display features for the preprocessed file
        i.save("path/to/new_file.wav")
        i.reset()  # Reset all changes and come back to the original file
    """

    @staticmethod
    def create_for(path, sr=None, duration=None):
        return AudioProcessingInterface(path, sr=sr, duration=duration)
    
    @staticmethod
    def create_from_data(path, y, sr=None):
        return AudioProcessingInterface(path, y=y, sr=sr)
    
    @staticmethod
    def create_from_mel(path, mel, sr=None):
        return AudioProcessingInterface(path, y=librosa.feature.inverse.mel_to_audio(mel), sr=sr)
    
    @staticmethod
    def create_from_log_mel(path, log_mel, sr=None):
        return AudioProcessingInterface(
            path,
            y=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(log_mel)),
            sr=sr
        )
        

    def __init__(self, path: str, y=None, sr=None, duration: Union[float, None]=None):
        """
        :param path: Path to the .wav audio file
        :param sr: Sample rate for the WAV file, None means the audio isn't resampled
        """
        self._path = path
        if y is None:
            self._y, self._sr = librosa.load(path, sr=sr, duration=duration)
        else:
            self._y, self._sr = y, sr or 22050

    def get_path(self) -> str:
        return self._path

    def get_data(self) -> tuple:
        return self._y, self._sr
    
    def extract_window(self, duration: float, method: str = "start") -> 'AudioProcessingInterface':
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
                self._y = self._y[:window_size]
            case "end":
                self._y = self._y[-window_size:]
            case "random":
                pass  # TODO
        return self

    def log_spectrogram(self):
        return librosa.amplitude_to_db(np.abs(librosa.stft(self._y)), ref=np.max)
    
    def mel_spectrogram(self, n_mels=128):
        return librosa.feature.melspectrogram(y=self._y, sr=self._sr, n_mels=n_mels)

    def log_mel_spectrogram(self, n_mels=128):
        return librosa.power_to_db(self.mel_spectrogram(n_mels))

    def mfcc(self, n_features=20, S=None):
        """
        :param n_features: Features resolution
        :param S: Precomputed spectogram (mel base)
        """
        if S is not None:
            return librosa.feature.mfcc(S=S, sr=self._sr, n_mfcc=n_features)
        return librosa.feature.mfcc(y=self._y, sr=self._sr, n_mfcc=n_features)    
    

    def display(self, qualifier: str="", n_mels=128, mfcc_features=20 ,save_to_dir=None, headless=True):
        fig, ax = plt.subplots(nrows=3, sharex=False)

        # Log Spectrogram
        spec = self.log_spectrogram()
        spec_img = librosa.display.specshow(spec, x_axis='time', y_axis='log', sr=self._sr, ax=ax[0])
        fig.colorbar(spec_img, ax=ax[0])
        ax[0].set(title="Log Spectrogram")

        # Mel Spectrogram
        spec = self.log_mel_spectrogram(n_mels=n_mels)
        spec_img = librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=self._sr, ax=ax[1])
        fig.colorbar(spec_img, ax=ax[1])
        ax[1].set(title="Log Mel Spectrogram")

        # MFCC Features
        mfcc = self.mfcc(n_features=mfcc_features, S=spec)
        mfcc_img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[2])
        fig.colorbar(mfcc_img, ax=ax[2])
        ax[2].set(title="MFCC Features")

        plt.suptitle(qualifier + " > " + path.basename(self._path))

         # enregistrer les spectogrammes
        if save_to_dir:
            if not os.path.exists(save_to_dir):
               os.makedirs(save_to_dir)
            save_path = os.path.join(save_to_dir, f"{path.basename(self._path)}_spectrogram.png")
            fig.savefig(save_path)

        # Afficher les graphiques
        if not headless  :
            fig.show()
            fig.waitforbuttonpress()

    
    def reconstruct_from_mel(self, n_mels=128) -> 'AudioProcessingInterface':
        """
        Simulates a reconstruction from a mel spectogram (Not in logarithmic scale)
        Might be expanded in the future to take in arbitrary spectrogram data depending on the NN output 
        """
        mel = self.mel_spectrogram(n_mels=n_mels)
        self._y = librosa.feature.inverse.mel_to_audio(mel, sr=self._sr)
        return self
    
    def reconstruct_from_mfcc(self, n_mels=128, mfcc_features=20) -> 'AudioProcessingInterface':
        """
        Simulates a reconstruction from mfcc features
        Might be expanded in the future to take in arbitrary mfcc data depending on the NN output
        """
        mfcc = self.mfcc(n_features=mfcc_features)
        self._y = librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=n_mels)
        self._y = librosa.feature.inverse.mel_to_audio(self._y)
        return self

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
        makedirs(path.dirname(outpath), exist_ok=True)
        sf.write(outpath, self._y, self._sr)
        return self
        