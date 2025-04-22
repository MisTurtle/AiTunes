import autoloader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from aitunes.audio_generation.simple_audio_streams import generate_sine_wave
from aitunes.audio_processing.processing_interface import AudioProcessingInterface

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
axes = np.array([
    fig.add_subplot(gs[0, :]),  # Wave form plot
    fig.add_subplot(gs[1, 0]),  # Spectrogram
    fig.add_subplot(gs[1, 1]),  # Log Spectrogram
    fig.add_subplot(gs[2, 0]),  # Mel Spectrogram
    fig.add_subplot(gs[2, 1])   # Log Mel Spectrogram
])

sr = 44100 * 10
out_path = generate_sine_wave("visuals", duration=2, hz=[512, 1024, 2048, 4096, 8192], sample_rate=sr, smoothing=False)
i = AudioProcessingInterface(out_path, mode='file', sr=sr)
i.extract_window(0.005, method='start').draw_wave(axes[0], color='b', alpha=0.75, label='Signal analogique')

sr = 22050
out_path = generate_sine_wave("visuals", duration=2, hz=[512, 1024, 2048, 4096, 8192], sample_rate=sr, smoothing=False)
i = AudioProcessingInterface(out_path, mode='file', sr=sr)
i.extract_window(0.005, method='start').draw_wave(axes[0], color='r', alpha=1, label='Signal numérique (22.05kHz)')
i.get_plot_for(
    ['spec', 'log_spec', 'mel', 'log_mel'],
    suptitle="",
    title=["Spectrogramme (linéaire)", "Spectrogramme (logarithmique)", "Spectrogramme Mel", "Spectrogramme Mel (logarithmique)"],
    n_mels=256, fig=fig, axes=axes[1:]
)

axes[0].legend()
axes[1].set_xlabel("")
axes[2].set_xlabel("")

fig.tight_layout()
plt.show()
