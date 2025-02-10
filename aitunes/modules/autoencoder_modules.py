import torch
import torch.nn as nn
from torchsummary import summary


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

    def __init__(self, encoder_dimensions, show_summary: bool = True):
        super().__init__()

        self._encoder = SimpleEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])
        if show_summary:
            summary(self, (encoder_dimensions[0], ))
    
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
    def __init__(self, encoder_dimensions, show_summary: bool = True):
        super().__init__()

        self._encoder = VariationalEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])
        
        if show_summary:
            summary(self, (encoder_dimensions[0], ))

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        return mu + log_var * epsilon

    def forward(self, x):
        mu, log_var = self._encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, self._decoder(z), mu, log_var


class CVAE(nn.Module):

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim, show_summary: bool = True):
        super(CVAE, self).__init__()
        
        self.input_shape = input_shape  # Data input shape (channels, height, width)
        self.conv_filters = conv_filters  # Number of channels to output after each layer
        self.conv_kernels = conv_kernels  # Convolutional kernel size
        self.conv_strides = list(map(lambda x: x if isinstance(x, tuple) else (x, x), conv_strides))  # How much to move the kernel after each iteration
        self.latent_space_dim = latent_space_dim  # To which dimension is the data compressed
        self.shape_at_i = []  # Padding to apply on ConvTranspose2d to consistently obtain an output the same size as the input (Because different input sizes produce the same output size)
        self._shape_before_bottleneck = self._calculate_shape_before_bottleneck()
        
        # Encoder and Decoder
        self._encoder = self._create_encoder()
        self._mu = nn.Linear(self._shape_before_bottleneck[0], self.latent_space_dim)
        self._log_var = nn.Linear(self._shape_before_bottleneck[0], self.latent_space_dim)
        self._decoder = self._create_decoder()

        if show_summary:
            summary(self, (*input_shape, ))


    def _create_encoder(self):
        layers = []
        input_channels = self.input_shape[0]

        # Couches convolutionnelles
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
            layers.append(nn.ReLU())  #non-linéarité
            layers.append(nn.BatchNorm2d(out_channels)) #Normalisation
            input_channels = out_channels
        
        layers.append(nn.Flatten())
        return nn.Sequential(*layers) 
    
    def _compute_decoder_padding(self, i) -> tuple[int, int]:
        """
        Takes the index of a convolutional layer and computes the padding to apply to its transpose to get the exact same output shape
        :param conv_index: The convolutional layer index
        :return: Symmetrical padding, one-sided padding
        """
        kernel_size, stride = self.conv_kernels[i], self.conv_strides[i]
        
        padding = kernel_size // 2

        h_in, w_in = self.shape_at_i[i]
        h_out = (h_in - 1) * stride[0] - 2 * padding + 1 * (kernel_size - 1) + 1
        w_out = (w_in - 1) * stride[1] - 2 * padding + 1 * (kernel_size - 1) + 1
        h_out_target, w_out_target = self.shape_at_i[i - 1] if i > 0 else self.input_shape[1:]
        p_h, p_w = h_out_target - h_out, w_out_target - w_out

        # Source: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        # Do not remove the following, in case things break again ;-;
        # print(f"Shape before {i} convolution should be {self.shape_at_i[i]}")
        # print("Output size:", h_out, w_out)
        # print("Target output size:", h_out_target, w_out_target)

        return padding, (p_h, p_w)
            
    
    def _create_decoder(self):
        layers = [nn.Linear(self.latent_space_dim, self._shape_before_bottleneck[0])]
        
        # Transformee le vecteur 1D en une forme 3D
        layers.append(nn.Unflatten(1, self._shape_before_bottleneck[1:]))

        input_channels = self.conv_filters[-1]
        for i in range(len(self.conv_filters) - 1, 0, -1):
            out_channels, kernel_size, stride = self.conv_filters[i - 1], self.conv_kernels[i], self.conv_strides[i]
            padding, (p_h, p_w) = self._compute_decoder_padding(i)
            
            layers.append(nn.ConvTranspose2d(input_channels, out_channels, kernel_size, stride, padding=padding, output_padding=(p_h, p_w)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            
            input_channels = out_channels

        padding, (p_h, p_w) = self._compute_decoder_padding(0)
        layers.append(nn.ConvTranspose2d(input_channels, self.input_shape[0], self.conv_kernels[0], self.conv_strides[0], padding=padding, output_padding=(p_h, p_w)))
        layers.append(nn.Sigmoid())
        # layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def _calculate_shape_before_bottleneck(self):
        """
        Calcule les dimensions exactes avant la couche de goulot d'étranglement, et la taille attendue pour chaque convolution inverse (pour calcul du padding)
        """
        h, w = self.input_shape[1:]
        
        for kernel_size, stride in zip(self.conv_kernels, self.conv_strides):
            # Compute the output size from the conv layer
            padding = kernel_size // 2
            h = (h + 2 * padding - kernel_size) // stride[0] + 1
            w = (w + 2 * padding - kernel_size) // stride[1] + 1

            self.shape_at_i.append((h, w))
        
        flattened_size = self.conv_filters[-1] * h * w
        return (flattened_size, self.conv_filters[-1], h, w)

    def _encode(self, x):
        x = self._encoder(x)
        mu, log_var = self._mu(x), self._log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        return mu + log_var * epsilon

    def _decode(self, z):
        return self._decoder(z)
    
    def forward(self, x):
        mu, log_var = self._encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self._decode(z)
        return z, x_reconstructed, mu, log_var
