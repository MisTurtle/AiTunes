import os
import numpy as np
from scipy.io.wavfile import write


def generate_sine_wave(to: str, sample_rate: int = 44100, duration: float = 5, hz: float = 440.0, amp: float = 0.5):
    # Adapted from https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
    assert -1 <= amp <= 1  # Amp has to be between -1 and 1 so the normalization to int16 works
    
    timepoints = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amp * np.sin(2 * np.pi * hz * timepoints)
    sine_wave_pcm = np.int16(sine_wave * 32767)  # Normalize to int16 which is the pcm data size (no compression)

    write(
        os.path.join(to, f"sine_wave_{int(hz)}.wav"),
        sample_rate, sine_wave_pcm
    )


def generate_ascending_sine_wave(to: str, sample_rate: int = 44100, duration: float = 5, hz_min: float = 10, hz_max: float = 500, amp: float = 0.5):
    # Adapted from https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
    assert -1 <= amp <= 1  # Amp has to be between -1 and 1 so the normalization to int16 works
    
    timepoints = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    hz = np.linspace(hz_min, hz_max, int(sample_rate * duration), endpoint=False)

    sine_wave = amp * np.sin(2 * np.pi * hz * timepoints)
    sine_wave_pcm = np.int16(sine_wave * 32767)  # Normalize to int16 which is the pcm data size (no compression)

    write(
        os.path.join(to, f"sine_asc_wave_{int(hz_min)}-{int(hz_max)}.wav"),
        sample_rate, sine_wave_pcm
    )


def generate_instrument_sound(to: str, sample_rate=44100, duration=3.0, frequency=261.63):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    harmonics = [1.0, 0.5, 0.3, 0.2, 0.1]
    signal = sum(harmonics[i] * np.sin(2 * np.pi * (i + 1) * frequency * t) for i in range(len(harmonics)))
    
    envelope = np.exp(-3 * t)
    signal = signal * envelope
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    
    write(
        os.path.join(to, f"instrument_{frequency}Hz.wav"),
        sample_rate, signal
    )
