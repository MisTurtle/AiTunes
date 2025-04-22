import autoloader
from os import path
path_to_audio = path.join("tests", "test_samples", "audio", "073087.mp3")


from aitunes.audio_processing import AudioProcessingInterface

i = AudioProcessingInterface.create_for(path_to_audio, mode="file", label="Données originales")
i2 = i.extract_window(10.0, method="start", label="Extrait de 10s")
i3 = i.copy(label="Extrait atténué").preprocess(lambda y: y * 0.5)
i.full_summary([i2, i3], n_mels=128, mfcc_features=13)
