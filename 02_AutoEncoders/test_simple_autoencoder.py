import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
        x = self._encoder(x)
        return self._decoder(x)


def plot_progress(values):
    ax.clear()
    ax.plot(values, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Live Plot)")
    ax.legend()
    plt.draw()
    plt.pause(0.1)


def example1():
    # This will generate 500'000 random 5-d vectors, where the 2nd and 4th dimension all depend on the other three.
    # The network will be trained to compress this data and uncompress it afterwards, trying to minimize the loss
    # This small example will allow to see if the system works for simple tasks like this one, in order to move on to more complex tasks afterwards
    
    # Generate the network
    model = SimpleAutoEncoder((5, 4, 3))
    loss_criterion = nn.MSELoss()  # Mean Squared Error loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 0.001 learning rate because it seems decent compared to past personal projects

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
        prediction = model(epoch_data)
        loss = loss_criterion(prediction, epoch_data)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        if (epoch + 1) % 10 == 0:
            plot_progress(loss_values)

    print(f"Trained in {time.time() - start_time:.3f}s. Final loss was {loss_values[-1]}")

    # Testing phase
    print("Testing on", test_data)
    model.eval()
    with torch.no_grad():
        test_prediction = model(test_data)
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
            result = model(vector)
            print(f"Test Case: \n\tInp: {vector.tolist()}\n\tOut: {result.tolist()}\n\tLoss: {loss_criterion(result, vector).item()}")
            

def main(): 
    example1()

if __name__ == "__main__":
    # Prepare for plotting 
    plt.ion()
    fig, ax = plt.subplots()
    main()
    plt.ioff()
    plt.close("all")


