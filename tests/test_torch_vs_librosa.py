from typing import Literal
import numpy as np
import torch
import autoloader
import librosa
from matplotlib import pyplot as plt
import torchaudio.transforms as T

from aitunes.audio_processing.processing_interface import AudioProcessingInterface

# Inspired from https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html

def plot_spectrogram(specgram, title=None, ylabel: Literal["freq_bin", "Hz"]="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    
    
    if ylabel == "Hz" and sample_rate and n_mels:
        mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sample_rate / 2)
        yticks = np.linspace(0, n_mels - 1, 6).astype(int)
        ylabels = [f"{int(mel_frequencies[i])}" for i in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_ylabel("Frequency (Hz)")
        
        img = ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")
    else:
        ax.set_ylabel(ylabel)
        img = ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


n_fft = 1024
hop_length = 512
n_mels = 128


i = AudioProcessingInterface(r"tests\test_samples\audio\073087.mp3", mode='file', n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
y, sample_rate = i.get_data()
y_tensor = torch.tensor(y).float().unsqueeze(0)

torch_mel_spec = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=None,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=n_mels,
    mel_scale="htk",
)
torch_log_mel_transform = T.AmplitudeToDB(stype="power", top_db=80)

from_torch = torch_mel_spec(y_tensor).cpu()
from_torch_log = torch_log_mel_transform(torch_mel_spec(y_tensor)).cpu()
from_librosa = i.mel_spectrogram(
    n_mels=n_mels,
    n_fft=n_fft,
    win_length=None,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney"
)

plot_spectrogram(from_torch[0], "PyTorch Mel Spectrogram")
plot_spectrogram(from_librosa, "Librosa Mel Spectrogram")
plot_spectrogram(from_torch_log[0], "PyTorch Log-Mel Spectrogram", ylabel="Hz")
lib_fig, _ = i.get_plot_for(['log_mel'], n_mels=n_mels)
plt.show()

