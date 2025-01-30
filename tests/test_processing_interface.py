import autoloader
from os import path, listdir
from aitunes.audio_processing import AudioProcessingInterface

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    samples_folder = path.join("assets", "Samples", "generated")
    output_folder = path.join("assets", "Samples", "processed")
    
    n_mels = 128
    mfcc_features = 64
    for filename in listdir(samples_folder):
        if path.isdir(path.join(samples_folder, filename)):
            continue

        i = AudioProcessingInterface.create_for(path.join(samples_folder, filename), mode="file")
        i.summary(n_mels=n_mels, mfcc_features=mfcc_features)
        i.extract_window(5, method="start")
        i.summary(qualifier="processed", n_mels=n_mels, mfcc_features=mfcc_features).save(path.join(output_folder, filename))

        i.reconstruct_from_mel(n_mels=n_mels).save(path.join(output_folder, "from_mel_" + filename))
        i.reconstruct_from_mfcc(n_mels=n_mels, mfcc_features=mfcc_features).save(path.join(output_folder, "from_mfcc_" + filename))

    plt.waitforbuttonpress()
    plt.close('all')
