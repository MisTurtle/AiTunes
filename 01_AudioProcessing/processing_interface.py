from os import path, listdir, mkdir
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


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
    def create_for(path):
        return AudioProcessingInterface(path)

    def __init__(self, path: str, sr=None):
        """
        :param path: Path to the .wav audio file
        :param sr: Sample rate for the WAV file, None means the audio isn't resampled
        """
        self._path = path
        self._y, self._sr = librosa.load(path, sr=None)

    def get_path(self) -> str:
        return self._path

    def get_data(self) -> tuple:
        return self._y, self._sr

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

    def display(self, qualifier: str="", n_mels=128, mfcc_features=20):
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
        self._y, self._sr = librosa.load(self._path, sr=None)
        return self
    
    def save(self, outpath: str) -> 'AudioProcessingInterface':
        if not path.exists(path.dirname(outpath)):
            mkdir(path.dirname(outpath))
        sf.write(outpath, self._y, self._sr)
        return self
        

if __name__ == '__main__':
    samples_folder = path.join("Samples", "generated")
    output_folder = path.join("Samples", "processed")
    
    for filename in listdir(samples_folder):
        i = AudioProcessingInterface.create_for(path.join(samples_folder, filename))
        i.display(n_mels=256, mfcc_features=128)
        i.preprocess(lambda y: np.multiply(y, 10))
        i.display(qualifier="processed", n_mels=256, mfcc_features=128)
        i.save(path.join(output_folder, filename))
        i.reset()

        # i.reconstruct_from_mel().save(path.join(output_folder, "built_from_mel.wav"))
        # i.reset().reconstruct_from_mfcc(mfcc_features=128).save(path.join(output_folder, "built_from_mfcc.wav"))
