from os import path, listdir
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
from skimage.io import imread
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim


def start():
    image_folder = path.join("tests", "test_samples", "image")
    files = listdir(image_folder)
    if len(files) == 0:
        return
    
    current_image_id = -1
    image = None

    def update(_=None):
        processed = image.copy()
        
        if slider_blur.val > 0:
            processed = gaussian(processed, slider_blur.val, truncate=2)
        if slider_noise.val > 0:
            processed = random_noise(processed, mode='s&p', rng=0, clip=True, amount=slider_noise.val)

        axes[1].imshow(processed)
        fig.suptitle(suptitle % ssim(image, processed, data_range=1.0, win_size=5, channel_axis=-1))
        plt.show()
    
    def next_image(_=None):
        nonlocal current_image_id, image
        current_image_id = (current_image_id + 1) % len(files)
        image = imread(path.join(image_folder, files[current_image_id])).astype(np.float32) / 255
        axes[0].imshow(image)
        update()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].set_title("Modified Image")
    axes[1].axis('off')
    suptitle = "Structural Similarity: %.3f"

    ax_blur = plt.axes([0.1, 0.02, 0.65, 0.03])
    ax_noise = plt.axes([0.1, 0.06, 0.65, 0.03])
    ax_next = plt.axes([0.80, 0.05, 0.1, 0.075])
    
    slider_blur = Slider(ax_blur, 'Blur', valmin=0, valmax=2, valstep=0.1, valinit=0)
    slider_noise = Slider(ax_noise, 'Noise', valmin=0, valmax=1, valstep=0.05, valinit=0)
    btn_next = Button(ax_next, "Next")
    
    slider_blur.on_changed(update)
    slider_noise.on_changed(update)
    btn_next.on_clicked(next_image)
    
    next_image()

if __name__ == "__main__":
    start()
