import os
import numpy as np
from scipy.io.wavfile import write


output_folder = "Samples/generated"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def generate_sine_wave(sample_rate: int = 44100, duration: float = 5, hz: float = 440.0, amp: float = 0.5):
    assert -1 <= amp <= 1  # Amp has to be between -1 and 1 so the normalization to int16 works
    
    timepoints = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amp * np.sin(2 * np.pi * hz * timepoints)
    sine_wave_pcm = np.int16(sine_wave * 32767)  # Normalize to int16 which is the pcm data size (no compression)

    write(
        os.path.join(output_folder, f"sine_wave_{int(hz)}.wav"),
        sample_rate, sine_wave_pcm
    )


def generate_ascending_sine_wave(sample_rate: int = 44100, duration: float = 5, hz_min: float = 10, hz_max: float = 500, amp: float = 0.5):
    assert -1 <= amp <= 1  # Amp has to be between -1 and 1 so the normalization to int16 works
    
    timepoints = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    hz = np.linspace(hz_min, hz_max, int(sample_rate * duration), endpoint=False)

    sine_wave = amp * np.sin(2 * np.pi * hz * timepoints)
    sine_wave_pcm = np.int16(sine_wave * 32767)  # Normalize to int16 which is the pcm data size (no compression)

    write(
        os.path.join(output_folder, f"sine_asc_wave_{int(hz_min)}-{int(hz_max)}.wav"),
        sample_rate, sine_wave_pcm
    )


if __name__ == "__main__":
    generate_sine_wave()
    generate_ascending_sine_wave(hz_min=150, hz_max=300)
    generate_ascending_sine_wave(hz_min=300, hz_max=150)