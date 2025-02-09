import autoloader
from os import path, listdir
from aitunes.audio_processing import AudioProcessingInterface
import matplotlib.pyplot as plt


if __name__ == '__main__':
    samples_folder = path.join("assets", "Samples", "generated")
    output_folder = path.join("assets", "Samples", "processed")
    
    n_mels = 128
    mfcc_features = 64
    for filename in listdir(samples_folder):
        if path.isdir(path.join(samples_folder, filename)):
            continue

        i = AudioProcessingInterface.create_for(path.join(samples_folder, filename), mode="file", label="Original")
        i2 = i.extract_window(1., method="start", label="Cropped")
        i3 = i.copy(label="Amplified").preprocess(lambda y: y * 1.2)
        i.full_summary([i2, i3], n_mels=n_mels, mfcc_features=mfcc_features, headless=False).save(path.join(output_folder, filename))
        
        fig, ax = i.compare_waves(i2, i3)
        ax.set(title="Fullscreen wave comparison")
        fig.show()
        fig.waitforbuttonpress()

        i2.reconstruct_from_mel(n_mels=n_mels).save(path.join(output_folder, "from_mel_" + filename))
        i2.reconstruct_from_mfcc(n_mels=n_mels, mfcc_features=mfcc_features).save(path.join(output_folder, "from_mfcc_" + filename))

        plt.close('all')
