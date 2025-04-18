import numpy as np
from scipy.io.wavfile import write

def simple_sine_wave(amp=0.5, duration=1, frequency=500, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = amp * np.sin(2 * np.pi * frequency * t)
    waveform_pcm = np.int16(waveform * 32767)
    write("out.wav", sample_rate, waveform_pcm)

simple_sine_wave()