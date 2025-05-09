import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

from matplotlib.widgets import Slider
from os import path

from aitunes.experiments import AutoencoderExperiment
from aitunes.utils import normalize, device


class MnistExperiment(AutoencoderExperiment):

    def __init__(self, model, weights_path, loss, optimizer):
        super().__init__("MNIST DIGITS", model, weights_path, loss, optimizer)
        transform = transforms.ToTensor()  # This will convert images to PyTorch tensors scaled to [0, 1] range for grayscale
        train_dataset = torchvision.datasets.MNIST(root=path.join("assets", "Samples"), train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=path.join("assets", "Samples"), train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=device))
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=device))
        self.embeds, self.labels = [], []
        # The following values are required for interactive evaluation and can either be set through the appropriate method
        # or automatically if a model evaluation is performed, through the prepare_embedded_region_mw middleware
        self.i_xmin, self.i_xmax = float('inf'), -float('inf')  # Bounding values for the x axis
        self.i_ymin, self.i_ymax = float('inf'), -float('inf')  # Bounding values for the y axis

        self.add_middleware(self.prepare_embedded_region_mw)

    @property
    def batch_size(self) -> int:
        return 32
    
    @property
    def batch_per_epoch(self) -> int:
        return int(math.ceil(len(self.train_loader) / self.batch_size))

    def next_batch(self, training, lookup_labels: bool = False):
        dataset = self.train_loader if training else self.test_loader
        
        for data, labels in dataset:  # Might use labels later
            data = data.to(device)
            if self.model.flatten:
                yield data.view(data.size(0), -1), labels
            else:
                yield data, labels
            
    def interactive_evaluation(self):
        # Interactive plot
        self.model.eval()
        with torch.no_grad():
            # Evaluate to build the latent space map
            if any(map(lambda x: math.isinf(x), [self.i_xmin, self.i_xmax, self.i_ymin, self.i_ymax])):
                self._support.log("/!\\ Interactive Evaluation for this experiment requires a model evaluation to be executed beforehand... Initiating...")
                self.evaluate()
            
            plt.ion()
            self.display_embed_plot()
            plt.ioff()

            fig, ax = plt.subplots(nrows=1, ncols=2)  # Left pane is the user input point, while the right pane is the display
            ax[0].set_xlim(self.i_xmin, self.i_xmax)
            ax[0].set_ylim(self.i_ymin, self.i_ymax)

            # 2D Point
            x, y = (self.i_xmin + self.i_xmax) / 2, (self.i_ymin + self.i_ymax) / 2
            point, = ax[0].plot(x, y, 'bo', markersize=10)
            # Sliders
            ax_x = plt.axes([0.1, 0.02, 0.65, 0.03])
            ax_y = plt.axes([0.1, 0.06, 0.65, 0.03])
            slider_x = Slider(ax_x, 'X', self.i_xmin, self.i_xmax, valinit=x)
            slider_y = Slider(ax_y, 'Y', self.i_ymin, self.i_ymax, valinit=y)
            
            def update(_):
                x_val, y_val = float(slider_x.val), float(slider_y.val)

                point.set_data([x_val], [y_val])
                if self.model.flatten:
                    decoder_input = torch.tensor([x_val, y_val], dtype=torch.float32)
                else:
                    decoder_input = torch.tensor([[x_val, y_val]], dtype=torch.float32)
                prediction = self.model.decode(decoder_input)
                rec_image = normalize(prediction.squeeze().cpu().numpy().reshape(28, 28)) * 255
                ax[1].imshow(rec_image.astype(np.uint8), cmap='gray')

            slider_x.on_changed(update)
            slider_y.on_changed(update)

            ax[0].set_title("Latent Dimension Pointer")
            ax[1].set_title("Latent Reconstruction")
            update(None)
            plt.show()


    def set_bounds(self, x: tuple[float, float], y: tuple[float, float]):
        self.i_xmin, self.i_xmax = min(x), max(x)
        self.i_ymin, self.i_ymax = min(y), max(y)
        
    def prepare_embedded_region_mw(self, og, pred, embeds, labels, args):
        self.embeds += embeds.tolist()
        self.labels += labels[0].tolist()
        self.i_xmin = min(self.i_xmin, torch.min(embeds[:, 0]).cpu().numpy())
        self.i_xmax = max(self.i_xmax, torch.max(embeds[:, 0]).cpu().numpy())
        self.i_ymin = min(self.i_ymin, torch.min(embeds[:, 1]).cpu().numpy())
        self.i_ymax = max(self.i_ymax, torch.max(embeds[:, 1]).cpu().numpy())
    
    def display_embed_plot(self):
        """
        This will only work exactly as expected if the latent dimension is 2D
        """
        _, ax = plt.subplots()
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_title(f"Latent Space (KL Weight = 0.001)")

        self.embeds = np.array(self.embeds)
        self.labels = np.array(self.labels)

        plt.scatter(self.embeds[:, 0], self.embeds[:, 1], c=self.labels, cmap='tab10')
        plt.colorbar()
        plt.show()
