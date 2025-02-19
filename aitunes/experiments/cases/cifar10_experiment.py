from matplotlib.widgets import Button
from aitunes.experiments import AutoencoderExperiment
from aitunes.utils import device

from os import path

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

class Cifar10Experiment(AutoencoderExperiment):

    def __init__(self, model, weights_path, loss, optimizer):
        super().__init__("CIFAR10", model, weights_path, loss, optimizer)
        transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=path.join("assets", "Samples", "CIFAR10"), train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=path.join("assets", "Samples", "CIFAR10"), train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, generator=torch.Generator(device=device))
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, generator=torch.Generator(device=device))
        self.embeds, self.labels = [], []
        
    def next_batch(self, training):
        dataset = self.train_loader
        if not training:
            dataset = self.test_loader
        
        for data, labels in dataset:  # Might use labels later
            data = data.to(device)
            if self.flatten:
                yield data.view(data.size(0), -1), labels
            else:
                yield data, labels
            
    def interactive_evaluation(self):
        # Interactive plot
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(self.test_loader)
            fig, axes = plt.subplots(3, self.test_loader.batch_size, figsize=(14, 6))

            def show_next(_):
                nonlocal data_iter
                try:
                    data, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.test_loader)
                    data, _ = next(data_iter)
                
                data = data.to(device)
                latent_sample, rec_img, *_ = self.model(data)
                
                
                for i in range(len(data)):
                    original_img = data[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                    reconstructed_img = rec_img[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                    latent_vector = latent_sample[i].cpu().numpy()
                    
                    axes[0, i].imshow(original_img)
                    axes[0, i].axis('off')
                    axes[1, i].imshow(reconstructed_img)
                    axes[1, i].axis('off')
                    axes[2, i].clear()
                    axes[2, i].bar(range(len(latent_vector)), latent_vector)
                
                axes[0, 0].set_title("Original Images")
                axes[1, 0].set_title("Reconstructed Images")
                axes[2, 0].set_title("Latent Space Representation")
                
                plt.show()
        
            plt.subplots_adjust(bottom=0.2)
            ax_next = plt.axes([0.45, 0.05, 0.1, 0.075])
            btn_next = Button(ax_next, "Next")
            btn_next.on_clicked(show_next)
            
            show_next(None)
