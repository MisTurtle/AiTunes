import torch
import numpy as np

from os import path, listdir
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage.io import imread
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure


torch_ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=7, reduction="sum")

def decolor(image, percent):
    gray = image.copy()
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gray = np.stack([gray, gray, gray], axis=-1)
    return percent * gray + (1 - percent) * image


def start():
    image_folder = path.join("tests", "test_samples", "image")
    files = listdir(image_folder)
    if len(files) == 0:
        return
    
    current_image_id = -1
    image = None

    def update(_=None):
        processed = image.copy()
        
        if slider_decolor.val > 0:
            processed = decolor(processed, slider_decolor.val)
        if slider_blur.val > 0:
            processed = gaussian(processed, slider_blur.val, truncate=2)
        if slider_noise.val > 0:
            processed = random_noise(processed, mode='s&p', rng=0, clip=True, amount=slider_noise.val)

        axes[1].imshow(processed)
        torch_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        torch_processed = torch.tensor(processed, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        fig.suptitle(suptitle % (
            ssim(image, processed, data_range=1.0, win_size=11, channel_axis=-1),
            torch_ssim(torch_image, torch_processed),
            torch.nn.functional.mse_loss(torch_image, torch_processed, reduce='mean')
        ))
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
    suptitle = "SSIM Tests\nScikit: %.5f  |  Torch: %.5f\nMSE Loss: %.5f"

    ax_decolor = plt.axes([0.1, 0.10, 0.65, 0.03])
    ax_blur = plt.axes([0.1, 0.06, 0.65, 0.03])
    ax_noise = plt.axes([0.1, 0.02, 0.65, 0.03])
    ax_next = plt.axes([0.80, 0.05, 0.1, 0.075])
    
    slider_decolor = Slider(ax_decolor, 'Decolor', valmin=0, valmax=1, valstep=0.05, valinit=0)
    slider_blur = Slider(ax_blur, 'Blur', valmin=0, valmax=2, valstep=0.1, valinit=0)
    slider_noise = Slider(ax_noise, 'Noise', valmin=0, valmax=1, valstep=0.05, valinit=0)
    btn_next = Button(ax_next, "Next")
    
    slider_blur.on_changed(update)
    slider_noise.on_changed(update)
    slider_decolor.on_changed(update)
    btn_next.on_clicked(next_image)
    
    next_image()

if __name__ == "__main__":
    start()
