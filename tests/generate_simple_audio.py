from os import path
import autoloader
from aitunes.audio_generation import *

if __name__ == "__main__":
    output_folder = path.join("assets", "Samples", "generated")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    generate_sine_wave(to=output_folder)
    generate_ascending_sine_wave(to=output_folder, hz_min=150, hz_max=300)
    generate_ascending_sine_wave(to=output_folder, hz_min=300, hz_max=150)
    generate_instrument_sound(to=output_folder)
