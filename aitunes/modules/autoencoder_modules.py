from typing import Union
import numpy as np
import torch
import torch.nn as nn
import aitunes.utils as utils
from torchsummary import summary


##################################
### Simple AutoEncoder Classes ###
##################################

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


class SimpleDecoder(nn.Module):

    def __init__(self, dimensions):
        """
        :param dimensions: Dimensions starting from the latent space size and going up until the output size
        """
        super().__init__()
        assert len(dimensions) > 0

        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i != len(dimensions) - 2:
                layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self._decoder(x)


class SimpleAutoEncoder(nn.Module):
    """
    A vanilla auto encoder class
    """

    def __init__(self, encoder_dimensions, decoder_activation: nn.Module = nn.Sigmoid, show_summary: bool = True):
        super().__init__()

        self._encoder = SimpleEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])
        self._activation = decoder_activation()
        if show_summary:
            summary(self, (encoder_dimensions[0], ), device=utils.device.type)
    
    def encode(self, input):
        return self._encoder(input)

    def decode(self, latent):
        return self._activation(self._decoder(latent))

    def forward(self, x):
        embedding = self.encode(x)
        return embedding, self.decode(embedding)


#######################################
### Variational AutoEncoder classes ###
#######################################

class VariationalEncoder(nn.Module):

    def __init__(self, dimensions):
        super().__init__()
        assert len(dimensions) > 1

        layers = []
        for i in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(nn.ReLU())
        self._encoder = nn.Sequential(*layers)
        
        self.mu = nn.Linear(dimensions[-2], dimensions[-1])  # Mu
        self.log_var = nn.Linear(dimensions[-2], dimensions[-1])  # Log of variance
    
    def forward(self, x):
        x = self._encoder(x)
        mu, log_var = self.mu(x), self.log_var(x)
        return mu, log_var


class VariationalAutoEncoder(nn.Module):
    """
    A variational auto encoder class
    """
    def __init__(self, encoder_dimensions, decoder_activation: nn.Module = nn.Sigmoid, show_summary: bool = True):
        super().__init__()

        self._encoder = VariationalEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])
        self._activation = decoder_activation()
        
        if show_summary:
            summary(self, (encoder_dimensions[0], ), device=utils.device.type)

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
    
    def encode(self, input):
        return self._encoder(input)

    def decode(self, z):
        return self._activation(self._decoder(z))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var
        

def compute_transpose_padding(kernel_size: int, stride: Union[int, tuple[int, int]], shape_at_i: list[int], i: int):
    kernel_size, stride = kernel_size, stride if isinstance(stride, tuple) else (stride, stride)
    padding = kernel_size // 2
    h_in, w_in = shape_at_i[i]
    h_out = (h_in - 1) * stride[0] - 2 * padding + 1 * (kernel_size - 1) + 1
    w_out = (w_in - 1) * stride[1] - 2 * padding + 1 * (kernel_size - 1) + 1
    h_out_target, w_out_target = shape_at_i[i + 1] if i < len(shape_at_i) - 1 else shape_at_i[-1]
    p_h, p_w = h_out_target - h_out, w_out_target - w_out
    print(f"Shape after {i + 1} transpose convolutions should be {(h_out_target, w_out_target)}")
    print("Output size:", h_out, w_out)
    print("Target output size:", h_out_target, w_out_target)
    print("Padding:", padding, " H:", p_h, "W:", p_w)
    return padding, (p_h, p_w)


##################################
### Convolutionnal VAE classes ###
##################################

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
            summary(self, (*input_shape, ), device=utils.device.type)


    def _create_encoder(self):
        layers = []
        input_channels = self.input_shape[0]

        # Couches convolutionnelles
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
            # layers.append(nn.BatchNorm2d(out_channels)) #Normalisation. Removed from the encoder because batch-wide normalization doesn't fit well with KL divergence
            layers.append(nn.ReLU())  #non-linéarité
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
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            
            input_channels = out_channels

        padding, (p_h, p_w) = self._compute_decoder_padding(0)
        layers.append(nn.ConvTranspose2d(input_channels, self.input_shape[0], self.conv_kernels[0], self.conv_strides[0], padding=padding, output_padding=(p_h, p_w)))
        layers.append(nn.Sigmoid())
        
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

    def encode(self, x):
        x = self._encoder(x)
        mu, log_var = self._mu(x), self._log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        sigma = torch.exp(0.5 * log_var)
        return mu + sigma * epsilon

    def decode(self, z):
        return self._decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var


#######################################################
### Convolutionnal VAE classes (v2, Smarter ResNet) ###
#######################################################


class ResnetDownsampler(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv2d(channels_in, channels_out // 2, kernel_size, stride=2, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels_out // 2)
        self.conv2 = nn.Conv2d(channels_out // 2, channels_out, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.conv_skipping = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=2)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        skipping = self.conv_skipping(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x) + skipping))
        return x


class ResnetUpsampler(nn.Module):
    """
    Simple ResNet UpSampler
    """

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, shape_at_i: list[tuple[int]], i: int):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.shape_at_i = shape_at_i
        self.i = i
        self.padding, self.output_padding = compute_transpose_padding(kernel_size, stride=2, shape_at_i=shape_at_i, i=i)

        self.conv1 = nn.ConvTranspose2d(channels_in, channels_in // 2, kernel_size, stride=2, padding=self.padding, output_padding=self.output_padding)
        self.bn1 = nn.BatchNorm2d(channels_in // 2)
        self.conv2 = nn.ConvTranspose2d(channels_in // 2, channels_out, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.conv_skipping = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=1, stride=2, output_padding=self.output_padding)
        self.activation = nn.ReLU()

    # def _compute_padding(self):
    #     kernel_size, stride = self.kernel_size, (2, 2)
    #     padding = kernel_size // 2
    #     h_in, w_in = self.shape_at_i[self.i]
    #     h_out = (h_in - 1) * stride[0] - 2 * padding + 1 * (kernel_size - 1) + 1
    #     w_out = (w_in - 1) * stride[1] - 2 * padding + 1 * (kernel_size - 1) + 1
    #     h_out_target, w_out_target = self.shape_at_i[self.i + 1]
    #     p_h, p_w = h_out_target - h_out, w_out_target - w_out
    #     # print(f"Shape after {self.i + 1} transpose convolutions should be {self.shape_at_i[self.i + 1]}")
    #     # print("Output size:", h_out, w_out)
    #     # print("Target output size:", h_out_target, w_out_target)

    #     return padding, (p_h, p_w)
    
    def forward(self, x):
        skipping = self.conv_skipping(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x) + skipping))
        return x
        

class ResNet2D(nn.Module):

    def __init__(self, input_size: tuple[int, ...], residual_blocks: int, residual_channels: int, latent_space_dim: int, activation_type: nn.Module = nn.Sigmoid, show_summary: bool = True):
        super().__init__()
        self.input_size = input_size
        self.residual_block_count = residual_blocks
        self.residual_channels = residual_channels
        self.latent_space_dim = latent_space_dim
        self.activation_type = activation_type
        self.shape_at_i = [self.input_size[1:]]

        self._encoder = self._create_encoder()
        self._decoder = self._create_decoder()
        
        self._mu = nn.Linear(np.prod(self._shape_before_bottleneck), self.latent_space_dim)
        self._log_var = nn.Linear(np.prod(self._shape_before_bottleneck), self.latent_space_dim)
        self._activation = activation_type()

        if show_summary:
            summary(self, (*input_size, ))

    def _create_encoder(self):
        encoder = nn.Sequential()
        encoder.add_module("ConvIn", nn.Conv2d(self.input_size[0], self.residual_channels, kernel_size=3, stride=2, padding=1))
        encoder.add_module("ConvInAct", self.activation_type())

        channels_in = self.residual_channels
        dummy = torch.tensor(np.random.random(self.input_size), dtype=torch.float32).unsqueeze(0)
        self.shape_at_i.append(encoder(dummy).shape[2:])
        for i in range(self.residual_block_count):
            encoder.add_module(f"ResBlock_Down{i}", ResnetDownsampler(channels_in, channels_in * 2, 3))
            self.shape_at_i.append(encoder(dummy).shape[2:])
            channels_in *= 2
        self._shape_before_bottleneck = encoder(dummy).shape[1:]

        encoder.add_module("Flatten", nn.Flatten())

        return encoder
    
    def _create_decoder(self) -> list[nn.Module]:
        decoder = nn.Sequential()
        decoder.add_module("FromLatent", nn.Linear(self.latent_space_dim, np.prod(self._shape_before_bottleneck)))
        decoder.add_module("Unflatten", nn.Unflatten(1, self._shape_before_bottleneck))

        channels_in = self.residual_channels * (2 ** self.residual_block_count)
        for i in range(self.residual_block_count):
            decoder.add_module(f"ResBlock_Up{i}", ResnetUpsampler(channels_in, channels_in // 2, 3, self.shape_at_i[::-1], i))
            channels_in //= 2
        
        padding, output_padding = compute_transpose_padding(3, 2, self.shape_at_i[::-1], self.residual_block_count)
        decoder.add_module("ToRecons", nn.ConvTranspose2d(self.residual_channels, self.input_size[0], kernel_size=3, stride=2, padding=padding, output_padding=output_padding))
        decoder.add_module("ActRecons", self.activation_type())
        return decoder
        
    def encode(self, x):
        x = self._encoder(x)
        return self._mu(x), self._log_var(x)
    
    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        sigma = torch.exp(0.5 * log_var)
        return mu + sigma * epsilon
    
    def decode(self, z):
        return self._decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var

# class VqVaeEncoder(nn.Module):

#     def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
#         self.input_shape = input_shape  # Data input shape (channels, height, width)
#         self.conv_filters = conv_filters  # Number of channels to output after each layer
#         self.conv_kernels = conv_kernels  # Convolutional kernel size
#         self.conv_strides = list(map(lambda x: x if isinstance(x, tuple) else (x, x), conv_strides))  # How much to move the kernel after each iteration
#         self.latent_space_dim = latent_space_dim  # To which dimension is the data compressed
#         self.shape_at_i = []  # Padding to apply on ConvTranspose2d to consistently obtain an output the same size as the input (Because different input sizes produce the same output size)
    
#         self.layers = self._create_layers()
    
#     def _create_layers(self):
#         layers = []
#         input_channels = self.input_shape[0]

#         # Couches convolutionnelles
#         for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
#             layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
#             layers.append(nn.ReLU())  #non-linéarité
#             input_channels = out_channels
        
#         layers.pop()  # Remove the last ReLU to have an unbounded latent space
#         layers.append(nn.Flatten())
#         return nn.Sequential(*layers) 

#     def forward(self, x):
#         return self.layers(x)


# class VqVaeQuantizer(nn.Module):

#     def __init__(self, n_embeddings, latent_space_dim, commitment_cost):
#         super().__init__()
#         self.n_embeddings = n_embeddings
#         self.latent_space_dim = latent_space_dim
#         self.commitment_cost = commitment_cost
    
#         self.embedding_table = nn.Embedding(n_embeddings, latent_space_dim)  # Embedding table (Codebook)
#         self.embedding_table.weight.data.uniform_(-1, 1)
    
#     def forward(self, x):
#         distance = ()


# class VqVae(nn.Module):
#     """
#     Ideally this would extend CVAE but the structure isn't modular enough
#     Let's allow some slack here and say repeating code is ok ^_^
#     """


