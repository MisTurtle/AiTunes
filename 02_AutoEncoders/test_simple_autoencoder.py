from os import path, makedirs
import time
from typing import Union
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image


class SimpleAutoEncoder(nn.Module):
    """
    A simple auto encoder class used to perform tests on small datasets
    """

    def __init__(self, encoder_dimensions):
        super().__init__()
        assert len(encoder_dimensions) > 0

        layers = []
        for i in range(len(encoder_dimensions) - 1):
            layers.append(nn.Linear(encoder_dimensions[i], encoder_dimensions[i + 1]))
            if i != len(encoder_dimensions) - 2:
                layers.append(nn.ReLU())
        self._encoder = nn.Sequential(*layers)

        layers = []
        for i in range(len(encoder_dimensions) - 1, 0, -1):
            layers.append(nn.Linear(encoder_dimensions[i], encoder_dimensions[i - 1]))
            if i > 1:
                layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        embedding = self._encoder(x)
        return embedding, self._decoder(embedding)
    

def plot_progress(values):
    ax.clear()
    ax.plot(values, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Live Plot)")
    ax.legend()
    plt.draw()
    plt.pause(0.1)


def normalize(nparray):
    return (nparray - np.min(nparray)) / (np.max(nparray) - np.min(nparray))


def example1(path_to_model: Union[str, None] = "02_AutoEncoders/ae_5d_vectors.pth"):
    # This will generate 500'000 random 5-d vectors, where the 2nd and 4th dimension all depend on the other three.
    # The network will be trained to compress this data and uncompress it afterwards, trying to minimize the loss
    # This small example will allow to see if the system works for simple tasks like this one, in order to move on to more complex tasks afterwards
    
    # Generate the network
    model = SimpleAutoEncoder((5, 4, 3))
    loss_criterion = nn.MSELoss()  # Mean Squared Error loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 0.001 learning rate because it seems decent compared to past personal projects

    def _(data_to_expand: np.array):
        """
        :param data_to_expand: Array of 3-d vectors to expand to 5-d vectors according to some mathematical relationship
        """
        if not isinstance(data_to_expand, np.ndarray):
            data_to_expand = np.array(data_to_expand)

        return np.column_stack((
            data_to_expand[:, 0],
            (data_to_expand[:, 0] + data_to_expand[:, 1] + data_to_expand[:, 2]) / 3,
            data_to_expand[:, 1],
            -data_to_expand[:, 2] - data_to_expand[:, 1] + 2 * data_to_expand[:, 0],
            data_to_expand[:, 2]
        ))

    # Produce training and test data
    data_size = 100000
    training_size = int(0.9 * data_size) 
    data = _(np.random.rand(data_size, 3))
    training_data, test_data = torch.tensor(data[:training_size, :], dtype=torch.float32), torch.tensor(data[training_size:, :], dtype=torch.float32)

    if path.exists(path_to_model):
        model.load_state_dict(torch.load(path_to_model, weights_only=True))
    else:
        epochs = 2000
        epoch_sample_size = 1000
        loss_values = []  # Keep track of loss evolution over time
        
        # Training phase
        print("Training on", training_data)
        start_time = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            epoch_indices = np.random.choice(training_data.shape[0], epoch_sample_size)
            epoch_data = training_data[epoch_indices]
            embedding, prediction = model(epoch_data)
            loss = loss_criterion(prediction, epoch_data)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

            if (epoch + 1) % 10 == 0:
                plot_progress(loss_values)

        print(f"Trained in {time.time() - start_time:.3f}s. Final loss was {loss_values[-1]}")

        torch.save(model.state_dict(), path_to_model)

    # Testing phase
    print("Testing on", test_data)
    model.eval()
    with torch.no_grad():
        embedding, test_prediction = model(test_data)
        test_loss = loss_criterion(test_prediction, test_data)
        print(f'Loss on testing data {test_loss.item()}')
    
    with torch.no_grad():
        while True:
            usr = input("Enter a 3 value vector (e.g. 0.3,0.2,0.1) or 'q' to exit:")
            
            if usr == 'q':
                break
            
            vector = list(map(lambda x: float(x), usr.split(",")))
            if len(vector) != 3:
                continue
            
            vector = torch.tensor(_([vector]), dtype=torch.float32)
            embedding, result = model(vector)
            print(f"Test Case: \n\tInp: {vector.tolist()}\n\tOut: {result.tolist()}\n\tLoss: {loss_criterion(result, vector).item()}")
            

def example2(path_to_model: Union[str, None] = "02_AutoEncoders/ae_mnist.pth"):
    # This will try to train an autoencoder on the MNIST dataset
    # Due to it being a simple autoencoder, the data should end up being overfitted and very little new generation value should be possible
    transform = transforms.ToTensor()  # This will convert images to PyTorch tensors scaled to [0, 1] range for grayscale
    train_dataset = torchvision.datasets.MNIST(root="Samples", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="Samples", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    latent_min, latent_max = float('inf'), -float('inf')

    # MNIST images are 28px by 28px, we'll try to compress MNIST images as low as to reach 2 dimensions
    model = SimpleAutoEncoder((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    loss_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if path.exists(path_to_model):
        model.load_state_dict(torch.load(path_to_model))
    else:
        print("Path to model weights could not be found... Training from scratch.")

        epochs = 20
        loss_values = []
        print("Training on the MNIST dataset...")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = .0
            for data, _ in train_loader:
                data = data.view(data.size(0), -1)  # Flatten images
                optimizer.zero_grad()
                embedding, prediction = model(data)
                loss = loss_criterion(prediction, data)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.size(0)
            
            avg_loss = epoch_loss / len(train_loader.dataset)
            loss_values.append(avg_loss)
            
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
            plot_progress(loss_values)

        print(f"Trained in {time.time() - start_time:.3f}s. Final loss was {loss_values[-1]}")
        torch.save(model.state_dict(), path_to_model)



    # Test the model
    model.eval()
    test_loss = .0
    showcase_n_elements = 10
    output_path = path.join("Samples", "processed", "mnist")
    makedirs(output_path, exist_ok=True)
    output_name = "ae_mnist_%d.png"
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1)
            embedding, prediction = model(data)

            # Update latent space min and max values
            latent_min = min(latent_min, torch.min(embedding))
            latent_max = max(latent_max, torch.max(embedding))

            loss = loss_criterion(prediction, data)
            test_loss += loss.item() * data.size(0)
            
            if showcase_n_elements > 0:
                # Save examples of reconstructed images
                case_path = path.join(output_path, output_name % showcase_n_elements)
                og_image = normalize(data[0].squeeze().cpu().numpy().reshape(28, 28)) * 255
                rec_image = normalize(prediction[0].squeeze().cpu().numpy().reshape(28, 28)) * 255
                
                final_image_np = (np.concatenate([og_image, rec_image], axis=1)).astype(np.uint8)
                final_image = Image.fromarray(final_image_np)
                final_image.save(case_path)

                print(f"Saved example at {case_path}")
                showcase_n_elements -= 1
            
        print(f"Loss on testing data {test_loss / len(test_loader.dataset)}")

    
    # Interactive plot
    fig, ax = plt.subplots(nrows=1, ncols=2)  # Left pane is the graph, while the right pane is the display
    ax[0].set_xlim(latent_min, latent_max)
    ax[0].set_ylim(latent_min, latent_max)

    # 2D Point
    mid = (latent_min + latent_max) / 2
    x, y = mid, mid
    point, = ax[0].plot(x, y, 'bo', markersize=10)

    # Sliders
    ax_x = plt.axes([0.1, 0.02, 0.65, 0.03])
    ax_y = plt.axes([0.1, 0.06, 0.65, 0.03])
    slider_x = Slider(ax_x, 'X', latent_min, latent_max, valinit=x)
    slider_y = Slider(ax_y, 'Y', latent_min, latent_max, valinit=y)
    
    def update(val):
        point.set_data([slider_x.val], [slider_y.val])
        prediction = model._decoder(torch.tensor([slider_x.val, slider_y.val], dtype=torch.float32))
        rec_image = normalize(prediction.squeeze().cpu().detach().numpy().reshape(28, 28)) * 255
        ax[1].imshow(rec_image.astype(np.uint8), cmap='gray')

    
    slider_x.on_changed(update)
    slider_y.on_changed(update)

    fig.show()
    while fig.waitforbuttonpress() is False:
        fig.waitforbuttonpress()


def main(): 
    example1()
    # example2()

if __name__ == "__main__":
    # Prepare for plotting 
    plt.ion()
    fig, ax = plt.subplots()
    main()
    plt.ioff()
    plt.close("all")
