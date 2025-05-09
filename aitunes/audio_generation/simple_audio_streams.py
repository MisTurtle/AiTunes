import os
import os.path as path
import random
import numpy as np

from typing import Union
from scipy.io.wavfile import write


def _get_smooth_amp(envelope_size: int, target_amp: float, fade_duration: int):
    """
    :param envelope_size: Size for the whole envelope (Audio timepoints)
    :param target_amp: The amplitude to fade in and out from
    :param fade_duration: How many timepoints it takes to fade in and out
    """
    fade_duration = min(envelope_size, fade_duration)
    envelope = np.ones(envelope_size)
    fade_in = np.linspace(0, 1, fade_duration)
    fade_out = np.linspace(1, 0, fade_duration)
    envelope[:fade_duration] = fade_in
    envelope[-fade_duration:] = fade_out
    return target_amp * envelope


def generate_sine_wave(to: str, sample_rate: int = 22050, duration: float = 5, hz: Union[float, list[float]] = 440.0, amp: float = 0.5, smoothing: bool = True):
    # Adapted from https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
    assert -1 <= amp <= 1  # Amp has to be between -1 and 1 so the normalization to int16 works
    
    # Timepoints to apply a sinewave to:
    timepoints = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if not isinstance(hz, list):
        hz = [hz]
    
    if smoothing:
        smooth_amp = _get_smooth_amp(len(timepoints), amp, int((duration / 20) * len(timepoints)))
    else:
        smooth_amp = np.ones_like(timepoints)
    # Generate sine wave
    sine_waves = [smooth_amp * np.sin(2 * np.pi * f * timepoints) for f in hz]
    sine_wave = sum(sine_waves) / len(hz)
    sine_wave_pcm = np.int16(sine_wave * 32767)  # Normalize to int16 which is the pcm data size (no compression)

    file_path = os.path.join(to, f"sinewave{len(hz)}.{'_'.join(map(lambda x: str(x), hz))}.wav")
        
    write(file_path, sample_rate, sine_wave_pcm)
    return file_path


def generate_ascending_sine_wave(to: str, sample_rate: int = 44100, duration: float = 5, hz_min: float = 10, hz_max: float = 500, amp: float = 0.5):
    # Adapted from https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
    assert -1 <= amp <= 1  # Amp has to be between -1 and 1 so the normalization to int16 works
    
    timepoints = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    hz = np.linspace(hz_min, hz_max, int(sample_rate * duration), endpoint=False)

    smooth_amp = _get_smooth_amp(len(timepoints), amp, int((duration / 20) * len(timepoints)))
    sine_wave = smooth_amp * np.sin(2 * np.pi * hz * timepoints)
    sine_wave_pcm = np.int16(sine_wave * 32767)  # Normalize to int16 which is the pcm data size (no compression)

    write(
        os.path.join(to, f"sineasc.{int(hz_min)}-{int(hz_max)}.wav"),
        sample_rate, sine_wave_pcm
    )

def generate_bouncing_wave(to: str, sample_rate: int = 44100, duration: float = 5, hz_min: float = 10, hz_max: float = 500, period: float = 1, amp: float = 0.5):
    timepoints = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq_modulation = hz_min + (hz_max - hz_min) * 0.5 * (1 + np.sin(2 * np.pi * timepoints / period))
    phase = 2 * np.pi * np.cumsum(freq_modulation) / sample_rate
    sine_wave = amp * np.sin(phase)
    sine_wave_pcm = np.int16(sine_wave * 32767)
    filename = os.path.join(to, f"sinebounce.{int(hz_min)}-{int(hz_max)}_{int(period)}s.wav")
    write(filename, sample_rate, sine_wave_pcm)

def generate_instrument_sound(to: str, sample_rate=44100, duration=3.0, frequency=261.63):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    harmonics = [1.0, 0.5, 0.3, 0.2, 0.1]
    signal = sum(harmonics[i] * np.sin(2 * np.pi * (i + 1) * frequency * t) for i in range(len(harmonics)))
    
    envelope = np.exp(-3 * t)
    signal = signal * envelope
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    
    write(
        os.path.join(to, f"instrument.{frequency}Hz.wav"),
        sample_rate, signal
    )


def generate_dataset_of_simple_instruments(to: str, sample_rate: int = 22050, unit_duration: float = 5.0, unit_per_type: int = 100):
    """
    Generates a dataset of simple audio files containing:
    - x simple sine waves
    - x combined sine waves
    - x ascending / descending sine waves
    """
    os.makedirs(to, exist_ok=True)
    for file in os.listdir(to):  # Clear any previous dataset
        os.remove(path.join(to, file))
    
    hz_range = 10, 4000
    r = lambda: random.randint(*hz_range)
    for _ in range(unit_per_type):
        generate_sine_wave(to, sample_rate=sample_rate, duration=unit_duration, hz=r())
    for _ in range(unit_per_type):
        generate_sine_wave(to, sample_rate=sample_rate, duration=unit_duration, hz=[r(), r()])
    for _ in range(unit_per_type):
        generate_sine_wave(to, sample_rate=sample_rate, duration=unit_duration, hz=[r(), r(), r()])
    for _ in range(unit_per_type):
        generate_bouncing_wave(to, sample_rate=sample_rate, duration=unit_duration, hz_min=r(), hz_max=r(), period=random.random() * unit_duration / 2)
    for _ in range(unit_per_type):
        generate_ascending_sine_wave(to, sample_rate=sample_rate, duration=unit_duration, hz_min=r(), hz_max=r())
