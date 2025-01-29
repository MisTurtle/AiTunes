from os import path, listdir
from aitunes.audio_processing import AudioProcessingInterface

import numpy as np


if __name__ == '__main__':
    samples_folder = path.join("Samples", "generated")
    output_folder = path.join("Samples", "processed")
    
    for filename in listdir(samples_folder):
        i = AudioProcessingInterface.create_for(path.join(samples_folder, filename))
        i.display(n_mels=256, mfcc_features=128)
        i.preprocess(lambda y: np.multiply(y, 10))
        i.display(qualifier="processed", n_mels=256, mfcc_features=128)
        i.save(path.join(output_folder, filename))
        i.reset()

        # i.reconstruct_from_mel().save(path.join(output_folder, "built_from_mel.wav"))
        # i.reset().reconstruct_from_mfcc(mfcc_features=128).save(path.join(output_folder, "built_from_mfcc.wav"))
