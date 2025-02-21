import autoloader
from os import path
from aitunes.audio_generation import *

if __name__ == "__main__":
    output_folder = path.join("tests", "audio", "test_samples")
    os.makedirs(output_folder, exist_ok=True)

    generate_sine_wave(to=output_folder)
    generate_sine_wave(to=output_folder, hz=[150, 330, 750])
    generate_ascending_sine_wave(to=output_folder, hz_min=150, hz_max=300)
    generate_ascending_sine_wave(to=output_folder, hz_min=300, hz_max=150)
    generate_instrument_sound(to=output_folder)
