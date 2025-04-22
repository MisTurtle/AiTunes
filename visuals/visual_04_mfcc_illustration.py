import autoloader
import matplotlib.pyplot as plt

from os import path
from aitunes.audio_processing.processing_interface import AudioProcessingInterface

file_path = path.join("tests", "test_samples", "audio", "073087.mp3")
i = AudioProcessingInterface(file_path, mode='file')
i.get_plot_for(
    ['wave', 'log_mel', 'mfcc'],
    title=['Waveform', 'Spectrogramme Mel (logarithmique)', 'Mel-Frequency Cepstral Coefficients'],
    mfcc_features=13
)
plt.show()
