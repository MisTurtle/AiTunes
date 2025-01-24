import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):

    def __init__(self, dimensions):
        """
        :param dimensions: Dimensions starting from the input size and going until the latent space size
        """
        super().__init__()
        assert len(dimensions) > 0

        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i != len(dimensions) - 2:
                layers.append(nn.ReLU())
        self._encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self._encoder(x)


class SimpleDecoder(SimpleEncoder):

    def __init__(self, dimensions):
        """
        :param dimensions: Dimensions starting from the latent space size and going up until the output size
        """
        super().__init__(dimensions)


class SimpleAutoEncoder(nn.Module):
    """
    A vanilla auto encoder class
    """

    def __init__(self, encoder_dimensions):
        super().__init__()

        self._encoder = SimpleEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])
    
    def forward(self, x):
        embedding = self._encoder(x)
        return embedding, self._decoder(embedding)


class VariationalEncoder(SimpleEncoder):

    def __init__(self, dimensions):
        super().__init__(dimensions[:-1])
        assert len(dimensions) > 1
        
        self.mu = nn.Linear(dimensions[-2], dimensions[-1])  # Mu
        self.log_var = nn.Linear(dimensions[-2], dimensions[-1])  # Log of variance
    
    def forward(self, x):
        x = super().forward(x)
        mu, log_var = self.mu(x), self.log_var(x)
        return mu, log_var


class VariationalAutoEncoder(nn.Module):
    """
    A variational auto encoder class
    """

    def __init__(self, encoder_dimensions):
        super().__init__()

        self._encoder = VariationalEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        return mu + log_var * epsilon

    def forward(self, x):
        mu, log_var = self._encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, self._decoder(z), mu, log_var
