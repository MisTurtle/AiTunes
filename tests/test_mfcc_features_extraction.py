import autoloader
import numpy as np
import librosa
from os import path, listdir
import matplotlib.pyplot as plt


def extract_mfcc(path, sr=None, nb_features=None):
    y, sr = librosa.load(path, sr=sr)  # Not specifying None will remap the sampling rate
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=nb_features or 20)

def extract_mel_spectrogram(path, sr=None, nb_mels=None):
    y, sr = librosa.load(path, sr=sr)
    return librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nb_mels or 20)

if __name__ == "__main__":
    # Inspired from https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html#librosa.feature.mfcc
    samples_folder = path.join("assets", "Samples", "generated")
    for filename in listdir(samples_folder):
        p = path.join(samples_folder, filename)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        
        S = extract_mel_spectrogram(p, nb_mels=128)
        S_img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax[0])

        mfcc = extract_mfcc(p, nb_features=128)
        mfcc_img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])

        fig.colorbar(S_img, ax=[ax[0]])
        ax[0].set(title=f'{filename} - Mel spectrogram')
        ax[0].label_outer()

        fig.colorbar(mfcc_img, ax=[ax[1]])
        ax[1].set(title=f'{filename} - MFCC')

        fig.show()
        fig.waitforbuttonpress()