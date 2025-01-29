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
    

class CVAE(nn.Module):

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):

        super(CVAE, self).__init__()
        self.input_shape = input_shape  
        self.conv_filters = conv_filters #le nombre de filtres dans chaque couche de convolution
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self._shape_before_bottleneck = self._calculate_shape_before_bottleneck()

        # Encoder and Decoder
        self.encoder = self._Encoder()
        self.decoder = self._Decoder()


    def _Encoder(self):
        layers = []

        input_channels = self.input_shape[0]

        # Couches convolutionnelles
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())  #non-linéarité
            layers.append(nn.BatchNorm2d(out_channels)) #Normalisation
            input_channels = out_channels

        layers.append(nn.Flatten())
        layers.append(nn.Linear(self._shape_before_bottleneck[0], self.latent_space_dim * 2))  # Sortie mu and log_var
        return nn.Sequential(*layers) 
    
    def _Decoder(self):

        layers = [nn.Linear(self.latent_space_dim, self._shape_before_bottleneck[0])]
        
        # Transformee le vecteur 1D en une forme 3D
        layers.append(nn.Unflatten(1, self._shape_before_bottleneck[1:]))

        input_channels = self.conv_filters[-1]
        for out_channels, kernel_size, stride in zip(
            reversed(self.conv_filters[:-1]),
            reversed(self.conv_kernels[:-1]),
            reversed(self.conv_strides[:-1])
        ):
            layers.append(nn.ConvTranspose2d(input_channels, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            input_channels = out_channels

        layers.append(nn.ConvTranspose2d(self.conv_filters[0], self.input_shape[0], self.conv_kernels[0], self.conv_strides[0], padding=1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    
    def _shape_before_bottleneck(self):
        """
        Calcule les dimensions exactes avant la couche de goulot d'étranglement.
        """
        h, w = self.input_shape[1:]
        for kernel_size, stride in zip(self.conv_kernels, self.conv_strides):
            h = (h - kernel_size + 2 * 1) // stride + 1
            w = (w - kernel_size + 2 * 1) // stride + 1
        return (self.conv_filters[-1] * h * w, self.conv_filters[-1], h, w)
    
    def _encode(self, x):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=1)
        return mu, log_var

    def _sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z):
        return self.decoder(z)
    
    
    def forward(self, x):
        mu, log_var = self._encode(x)
        z = self._sample(mu, log_var)
        x_reconstructed = self._decode(z)
        return z,x_reconstructed, mu, log_var

    

    
