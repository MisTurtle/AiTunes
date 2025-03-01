import math
import torch
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from aitunes.experiments import AutoencoderExperiment

class LinearExperiment(AutoencoderExperiment):

    @staticmethod
    def expand_3d_vector(val: np.array):
        if not isinstance(val, np.ndarray):
            val = np.array(val)    
        return np.column_stack((
            val[:, 0]**2,                     # Quadratic term
            np.sin(val[:, 1] * val[:, 2]),    # Sine of product
            np.log(1 + np.abs(val[:, 0])),    # Log transform
            val[:, 1] * val[:, 2],            # Interaction term
            np.exp(val[:, 2]) / np.exp(1)     # Exponential
        ))
    
    def __init__(self, model, weights_path, loss, optimizer):
        super().__init__("5D VECTORS", model, weights_path, loss, optimizer)
        self.training_data = self.expand_3d_vector(np.random.rand(1000, 3))
        self.evaluation_data = self.expand_3d_vector(np.random.rand(50, 3))

    @property
    def batch_size(self) -> int:
        return 32
    
    @property
    def batch_per_epoch(self) -> int:
        return int(math.ceil(self.training_data.shape[0] / self.batch_size))
    
    def next_batch(self, training):
        dataset = self.training_data if training else self.evaluation_data

        np.random.shuffle(dataset)
        for i in range(dataset.shape[0] // self.batch_size):
            yield torch.tensor(dataset[i * self.batch_size:(i + 1) * self.batch_size], dtype=torch.float32), None
        
    def interactive_evaluation(self):
        self.model.eval()
        with torch.no_grad():
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

            # Bar plot X labels and ticks
            X_labels = [f"Z[{i}]" for i in range(5)]
            X_ticks = np.arange(len(X_labels))
            # Create bar plots
            bar_og = ax.bar(X_ticks - 0.1, np.zeros_like(X_ticks), 0.2, label = "Original", color='b')
            bar_rec = ax.bar(X_ticks + 0.1, np.zeros_like(X_ticks), 0.2, label = "Reconstructed", color='r')
            # Add ticks
            ax.set_xticks(X_ticks, X_labels)
            ax.set_ylim(-0.1, 1.1)
            
            # Sliders
            plt.subplots_adjust(left=0.30)
            ax_1 = plt.axes([0.05, 0.05, 0.02, 0.90])
            ax_2 = plt.axes([0.10, 0.05, 0.02, 0.90])
            ax_3 = plt.axes([0.15, 0.05, 0.02, 0.90])
            slider_1 = Slider(ax_1, 'X1', 0, 1, valinit=0.75, orientation="vertical")
            slider_2 = Slider(ax_2, 'X2', 0, 1, valinit=0.50, orientation="vertical")
            slider_3 = Slider(ax_3, 'X3', 0, 1, valinit=0.25, orientation="vertical")
            
            def update(_=None):
                slider_data = np.array([[slider_1.val, slider_2.val, slider_3.val]], dtype=np.float32)
                og_heights = LinearExperiment.expand_3d_vector(slider_data)
                latent, rec_heights, *_ = self.model(torch.tensor(og_heights, dtype=torch.float32), training=False)
                rec_heights = rec_heights.cpu().numpy()

                for bar, height in zip(bar_og, og_heights[0]):
                    bar.set_height(height)
                
                for bar, height in zip(bar_rec, rec_heights[0]):
                    bar.set_height(height)
                
                fig.canvas.draw_idle()
            
            slider_1.on_changed(update)
            slider_2.on_changed(update)
            slider_3.on_changed(update)

            update()

            # Add plot labels
            ax.legend()
            ax.set_title("Linear Augmentation from 3D to 5D")
            ax.set_xlabel("Coordinate")
            ax.set_ylabel("Amplitude")
            ax.legend()
            
            plt.show()

