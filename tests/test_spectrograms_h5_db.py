import autoloader
import unittest
import shutil

from os import makedirs, path
from aitunes.utils.audio_utils import precompute_spectrograms_for_audio_folder, LowResolutionAudioFeatures


class TestSpectrogramsAsH5(unittest.TestCase):

    def setUp(self):
        self.input_dir = path.join("tests", "audio", "test_samples")
        self.output_dir = path.join("tests", "output", "spectrogram_databases")
        shutil.rmtree(self.output_dir)
        makedirs(self.output_dir, exist_ok=True)

    def test_write_to_database(self):
        quality = LowResolutionAudioFeatures(duration=5.0)
        precompute_spectrograms_for_audio_folder(
            audio_folder=self.input_dir,
            training_output=path.join(self.output_dir, "training.h5"),
            evaluation_output=path.join(self.output_dir, "evaluation.h5"),
            evaluation_proportions=0.05,
            features=quality,
            audio_preprocessing=lambda y: y,
            spec_preprocessing=lambda y: y,
            flush_every=5
        )

if __name__ == "__main__":
    unittest.main()
