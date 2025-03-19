import math
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from os import path
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

from aitunes.experiments import AutoencoderExperiment
from aitunes.utils import device


class Cifar10Experiment(AutoencoderExperiment):

    def __init__(self, model, weights_path, loss, optimizer):
        super().__init__("CIFAR10", model, weights_path, loss, optimizer)
        # transform = transforms.ToTensor()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])])
        self.train_dataset = torchvision.datasets.CIFAR10(root=path.join("assets", "Samples", "CIFAR10"), train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root=path.join("assets", "Samples", "CIFAR10"), train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=device))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=device))
        self.embeds, self.labels = [], []

    @property
    def batch_size(self) -> int:
        return 16
    
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
            train_i, test_i = 0, 0
            showing_test_dataset = True
            latent_sample, show_generated = None, False
            
            fig = plt.figure(figsize=(10, 7.5))
            gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1])
            axes = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, :])
            ]
            axes[0].set_title("Original Image")
            axes[1].set_title("Reconstructed Image")
            axes[2].set_title("Latent Space State")

            suptitle = "CIFAR10 Reconstruction Evaluation (Loss: %.3f)"
            fig.subplots_adjust(bottom=0.2)
            
            def show(move_by: int = 0):
                nonlocal test_i, train_i, latent_sample, show_generated
                
                if move_by != 0 and showing_test_dataset:
                    show_generated = False
                    test_i = (test_i + move_by) % len(self.test_dataset)
                elif move_by != 0 and not showing_test_dataset:
                    show_generated = False
                    train_i = (train_i + move_by) % len(self.train_dataset)
                
                if not show_generated:
                    dataset = self.test_dataset if showing_test_dataset else self.train_dataset
                    data, _ = dataset[test_i if showing_test_dataset else train_i]
                    data = data.to(device).unsqueeze(0)
                    latent_sample, rec_img, *args = self.model(data, training=False)
                    
                    loss = self._loss_criterion(rec_img, data, *args)
                    fig.suptitle(suptitle % loss[0])

                    
                    original_img = data[0].cpu().permute(1, 2, 0).numpy()
                    original_img = np.clip(original_img, -0.5, 0.5) + 0.5
                    axes[0].imshow(original_img)
                    axes[0].axis('off')
                else:
                    rec_img = self.model.decode(latent_sample)
                    fig.suptitle("Generated Image")
                    axes[0].clear()

                reconstructed_img = rec_img[0].cpu().permute(1, 2, 0).numpy()
                reconstructed_img = np.clip(reconstructed_img, -0.5, 0.5) + 0.5
                latent_vector = latent_sample[0].cpu().numpy()

                axes[1].imshow(reconstructed_img)
                axes[1].axis('off')
                axes[2].clear()
                # axes[2].bar(range(len(latent_vector)), latent_vector)
                plt.show()
            
            def switch(_):
                nonlocal showing_test_dataset, show_generated
                show_generated = False
                showing_test_dataset = not showing_test_dataset
                btn_switch.label.set_text("Current:\n" + ("Test" if showing_test_dataset else "Training") + " Dataset")
                show()
            
            def generate(_):
                nonlocal latent_sample, show_generated
                show_generated = True
                latent_sample = self.model.sample(2)
                show()

            ax_prev = plt.axes([0.24, 0.05, 0.1, 0.075])
            ax_next = plt.axes([0.38, 0.05, 0.1, 0.075])
            ax_switch = plt.axes([0.52, 0.05, 0.1, 0.075])
            ax_gen = plt.axes([0.66, 0.05, 0.1, 0.075])

            btn_prev = Button(ax_prev, "Prev")
            btn_prev.on_clicked(lambda _: show(-1))
            btn_next = Button(ax_next, "Next")
            btn_next.on_clicked(lambda _: show(1))
            btn_switch = Button(ax_switch, "Current:\nTest Dataset")
            btn_switch.on_clicked(switch)
            btn_gen = Button(ax_gen, "Generate")
            btn_gen.on_clicked(generate)

            show()
