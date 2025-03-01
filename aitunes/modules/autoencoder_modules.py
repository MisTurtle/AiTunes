from abc import ABC, abstractmethod
from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AiTunesAutoencoderModule(nn.Module, ABC):

    def __init__(self, input_shape: tuple[float], flatten: bool):
        """
        Base class for autoencoder modules in the AiTunes project. This allows for more abstract
        representations in other systems like experiments and interactive evaluations

        Args:
            input_shape (tuple[float]): Shape of the input given to the model
            flatten (bool): Whether the model expectes flattened, 1D input
        """
        super().__init__()
        self.input_shape = input_shape
        self._latent_shape = None
        self.flatten = flatten
    
    @property
    def latent_shape(self) -> tuple[int]:
        if self._latent_shape is None:
            self._latent_shape = self.encode(torch.rand((1, *self.input_shape))).shape[1:]
        return self._latent_shape

    @abstractmethod
    def forward(self, x, training):
        pass

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass


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


class SimpleAutoEncoder(AiTunesAutoencoderModule):
    """
    A vanilla auto encoder class
    """

    def __init__(self, encoder_dimensions, decoder_activation: nn.Module = nn.Sigmoid):
        super().__init__()

        self._encoder = SimpleEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])
        self._activation = decoder_activation()
    
    def encode(self, x):
        return self._encoder(x)

    def decode(self, latent):
        return self._activation(self._decoder(latent))

    def forward(self, x, training):
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
    def __init__(self, encoder_dimensions, decoder_activation: nn.Module = nn.Sigmoid):
        super().__init__()

        self._encoder = VariationalEncoder(encoder_dimensions)
        self._decoder = SimpleDecoder(encoder_dimensions[::-1])
        self._activation = decoder_activation()
    
    def encode(self, x):
        return self._encoder(x)

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
    
    def decode(self, z):
        return self._activation(self._decoder(z))

    def forward(self, x, training):
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
    # print(f"Shape after {i + 1} transpose convolutions should be {(h_out_target, w_out_target)}")
    # print("Output size:", h_out, w_out)
    # print("Target output size:", h_out_target, w_out_target)
    # print("Padding:", padding, " H:", p_h, "W:", p_w)
    return padding, (p_h, p_w)


##################################
### Convolutionnal VAE classes ###
##################################

class CVAE(AiTunesAutoencoderModule):

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        super().__init__(input_shape=input_shape, flatten=False)
        
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
    
    def forward(self, x, training):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var


###################################################
### Convolutionnal VAE classes (Smarter ResNet) ###
###################################################
# Inspired from https://github.com/LukeDitria/CNN-VAE/blob/master/RES_VAE.py

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
    
    def forward(self, x):
        skipping = self.conv_skipping(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x) + skipping))
        return x
        

class ResNet2D(AiTunesAutoencoderModule):

    def __init__(self, input_size: tuple[int, ...], residual_blocks: int, residual_channels: int, latent_space_dim: int, activation_type: Union[nn.Module, tuple[nn.Module, nn.Module]] = nn.Sigmoid):
        super().__init__(input_shape=input_size, flatten=False)
        self.residual_block_count = residual_blocks
        self.residual_channels = residual_channels
        self.latent_space_dim = latent_space_dim
        self.activation_type = (activation_type, activation_type) if not isinstance(activation_type, tuple) else activation_type
        self.shape_at_i = [self.input_size[1:]]

        self._encoder = self._create_encoder()
        self._decoder = self._create_decoder()
        
        self._mu = nn.Linear(np.prod(self._shape_before_bottleneck), self.latent_space_dim)
        self._log_var = nn.Linear(np.prod(self._shape_before_bottleneck), self.latent_space_dim)
        self._activation = activation_type()

    def _create_encoder(self):
        encoder = nn.Sequential()
        encoder.add_module("ConvIn", nn.Conv2d(self.input_size[0], self.residual_channels, kernel_size=3, stride=2, padding=1))
        if self.activation_type[0] is not None:
            encoder.add_module("ConvInAct", self.activation_type[0]())

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
        if self.activation_type[1] is not None:
            decoder.add_module("ActRecons", self.activation_type[1]())
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
    
    def forward(self, x, training):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var
    
    
########################################
### Even smarter 2d residual network ###
########################################

class ResidualStackV2(nn.Module):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
        A ResidualStack performing convolutions with residual connections on the input data.

        Args:
            num_hiddens (int): Number of convolution channels given as an input to the stack
            num_residual_layers (int): Number of residual connections to create
            num_residual_hiddens (int): Number of convolution channels to compress to and back from
        """
        super().__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._layers = []
        self._build_layers()
    
    def _build_layers(self):
        for i in range(self._num_residual_layers):
            conv3 = nn.Conv2d(
                in_channels=self._num_hiddens,
                out_channels=self._num_residual_hiddens,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            )
            conv1 = nn.Conv2d(
                in_channels=self._num_residual_hiddens,
                out_channels=self._num_hiddens,
                kernel_size=(1, 1)
            )
            self._layers.append(nn.Sequential(
                nn.ReLU(),
                conv3,
                nn.ReLU(),
                conv1
            ))
        self._layers = nn.ModuleList(self._layers)
    
    def forward(self, x):
        h = x
        for residual in self._layers:
            h = h + residual(h)
        return torch.relu(h)


class ResidualEncoderV2(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens):
        """
        Encoder module for a residual network.

        Args:
            in_channels (_type_): Number of channels the input values have
            num_hiddens (int): Number of convolution channels to reach within the encoder
            num_downsampling_layers (int): Number of downsampling layers to create in the Encoder. Each layer basically divides the input shape by 2. Min: 2 layers
            num_residual_layers (int): Number of residual connections to create in the Encoder and Decoder ResidualStacks
            num_residual_hiddens (int): Number of convolution channels to compress to in the ResidualStacks
        """
        super().__init__()
        self._in_channels = in_channels
        self._num_hiddens = num_hiddens
        self._num_downsampling_layers = num_downsampling_layers
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._convolution, self._residual_stack = None, None
        self._build_modules()
    
    def _build_modules(self):
        layers = []
        in_channels = self._in_channels 
        for i in range(self._num_downsampling_layers):
            if i == 0:
                out_channels = self._num_hiddens // 2
            elif i == 1:
                (in_channels, out_channels) = (self._num_hiddens // 2, self._num_hiddens)
            else:
                (in_channels, out_channels) = (self._num_hiddens, self._num_hiddens)
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(
            in_channels=self._num_hiddens,
            out_channels=self._num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        ))
        self._convolution = nn.Sequential(*layers)
        self._residual_stack = ResidualStackV2(self._num_hiddens, self._num_residual_layers, self._num_residual_hiddens)
    
    def forward(self, x):
        return self._residual_stack(self._convolution(x))


class ResidualDecoderV2(nn.Module):

    def __init__(self, out_channels, embedding_dim, num_hiddens, num_upsampling_layers, num_residual_layers, num_residual_hiddens, activation_type: Union[None, nn.Module] = None):
        """
        Decoder module for a residual network.

        Args:
            out_channels (_type_): Number of channels expected in the output
            embedding_dim (int): Embedding vectors dimension
            num_embeddings (int): Number of discrete embedding vectors
            num_upsampling_layers (int):  Number of upsampling layers to create. Min: 2 layers
            num_residual_layers (int): Number of residual connections to create in the ResidualStack
            num_residual_hiddens (int): Number of convolution channels to compress to in the ResidualStack
            activation_type (Union[None, nn.Module], optional): Activation function out of the decoder. If None, no activation is applied. Defaults to None. Defaults to None.
        """
        super().__init__()
        self._out_channels = out_channels
        self._embedding_dim = embedding_dim
        self._num_hiddens = num_hiddens
        self._num_upsampling_layers = num_upsampling_layers
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._activation_type = activation_type
        self._conv_in, self._upconv, self._residual_stack = None, None, None
        self._build_modules()
    
    def _build_modules(self):
        self._conv_in = nn.Conv2d(
            in_channels=self._embedding_dim,
            out_channels=self._num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self._residual_stack = ResidualStackV2(self._num_hiddens, self._num_residual_layers, self._num_residual_hiddens)
        layers = []
        for i in range(self._num_upsampling_layers):
            if i < self._num_upsampling_layers - 2:
                (in_channels, out_channels) = (self._num_hiddens, self._num_hiddens)
            elif i == self._num_upsampling_layers - 2:
                (in_channels, out_channels) = (self._num_hiddens, self._num_hiddens // 2)
            else:
                (in_channels, out_channels) = (self._num_hiddens // 2, self._out_channels)
            layers.append(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            if i != self._num_upsampling_layers - 1:
                layers.append(nn.ReLU())

        if self._activation_type is not None:
            layers.append(self._activation_type())

        self._upconv = nn.Sequential(*layers)
    
    def forward(self, x):
        h = self._residual_stack(self._conv_in(x))
        return self._upconv(h)


#########################
### VQ VAE COMPONENTS ###
#########################

class SonnetEMA(nn.Module):

    def __init__(self, decay, shape):
        """
        Sonnet Exponential Moving Average module tracking a moving average value

        Args:
            decay (float): Weight given to past values. Closer to one means a smoother and slower reaction to changes.
            shape (tuple[int]): Moving average shape
        """
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer('hidden', torch.zeros(*shape))
        self.register_buffer('average', torch.zeros(*shape))
    
    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)
    
    def forward(self, x):
        self.update(x)
        return self.average


class VectorQuantizer(nn.Module):

    def __init__(self, embedding_dim: int, num_embeddings: int, ema: bool = True, decay: float = 0.99, epsilon: float = 1e-5):
        """
        VectorQuantizer takes an input vector and maps it to the closest in the Embedding Table, performing an Exponential Moving Average iteration if enabled

        Args:
            embedding_dim (int): Embedding vectors dimension
            num_embeddings (int): Number of discrete embedding vectors
            ema (bool, optional): Whether to use Sonnet's Exponential Moving Average to prevent codebook collapse. Defaults to True.
            decay (float, optional): Weight given to past values. Closer to one means a smoother and slower reaction to changes. Defaults to 0.99.
            epsilon (float, optional): Epsilon to prevent numerical instability. Defaults to 1e-5.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.ema = ema

        # Vector Embedding Table
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embeddings.weight.data.uniform_(-1.7, 1.7)

        # EMA var
        self.EMA_cluster_counts = SonnetEMA(decay, (num_embeddings, ))
        self.EMA_embeddings = SonnetEMA(decay, self.embeddings.weight.shape[::-1])

        # Last loss value (retrieved as properties so it doesn't break torchsummary)
        self.last_codebook_loss = torch.tensor(0.0, requires_grad=False)
        self.last_commitment_loss = torch.tensor(0.0, requires_grad=True)
    
    def forward(self, x, training):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)  # squared input
            - 2 * flat_x @ self.embeddings.weight.T  # dot product with embeddings (transpose of weight)
            + (self.embeddings.weight ** 2).sum(1, keepdim=True).T  # squared embeddings
        )
    
        # Lookup tensors in the embedding lookup table
        encoding_indices = distances.argmin(1)
        quantized = self.embeddings(encoding_indices.view(x.shape[0], *x.shape[2:])).permute(0, 3, 1, 2)

        # First loss component to return
        self.last_codebook_loss = None if self.ema else torch.mean((quantized - x.detach()) ** 2)  # Quantized vectors should be close to encoder output
        self.last_commitment_loss = torch.mean((x - quantized.detach()) ** 2)  # Encoder output should be close to quantized vectors
        quantized = x + (quantized - x).detach()

        if self.ema and training:
            with torch.no_grad():
                encodings = F.one_hot(encoding_indices, num_classes=self.num_embeddings).float()  # Zeros everywhere, except for a 1 at the index of the table index
                # Update cluster count EMA
                cluster_count = encodings.sum(dim=0)
                updated_ema_cluster_counts = self.EMA_cluster_counts(cluster_count)
                # Update embeddings EMA
                embed_sums = torch.matmul(flat_x.transpose(0, 1), encodings)
                updated_ema_dw = self.EMA_embeddings(embed_sums)
                # Update embedding table
                n = updated_ema_cluster_counts.sum()
                updated_ema_cluster_counts = ((updated_ema_cluster_counts + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
                normalised_updated_ema_w = updated_ema_dw / updated_ema_cluster_counts.unsqueeze(0)
                self.embeddings.weight.copy_(normalised_updated_ema_w.t())
        
        return quantized

class VQ_ResNet2D(AiTunesAutoencoderModule):

    def __init__(
            self,
            input_shape,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
            embedding_dim,
            num_embeddings,
            decoder_activation: Union[None, nn.Module] = None,
            use_ema: bool = True,
            decay: float = 0.99,
            epsilon: float = 1e-5
    ):
        """
        Implementation of a VQVAE Residual Neural Network inspired by the original paper at https://arxiv.org/abs/1711.00937,
        Google Deepmind's implementation at https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
        and the port to PyTorch at https://github.com/airalcorn2/vqvae-pytorch/blob/master/vqvae.py

        Args:
            input_shape (tuple[int]): Size of the input passed to the model, as (channels, height, width). Height and Width are only used to display the summary
            num_hiddens (int): Number of convolution channels to reach within the encoder
            num_downsampling_layers (int): Number of downsampling layers to create in the Encoder. Each layer basically divides the input shape by 2. Min: 2 layers
            num_residual_layers (int): Number of residual connections to create in the Encoder and Decoder ResidualStacks
            num_residual_hiddens (int): Number of convolution channels to compress to in the ResidualStacks
            embedding_dim (int): Embedding vectors dimension
            num_embeddings (int): Number of discrete embedding vectors
            decoder_activation (Union[None, nn.Module], optional): Activation function out of the decoder. If None, no activation is applied. Defaults to None.
            use_ema (bool, optional): Whether to use Sonnet's Exponential Moving Average to prevent codebook collapse. Defaults to True.
            decay (float, optional): Weight given to past values. Closer to one means a smoother and slower reaction to changes. Defaults to 0.99.
            epsilon (float, optional): Epsilon to prevent numerical instability. Defaults to 1e-5.
        """
        super().__init__(input_shape=input_shape, flatten=False)

        self._encoder = ResidualEncoderV2(input_shape[0], num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1)
        self._vq = VectorQuantizer(embedding_dim, num_embeddings, use_ema, decay, epsilon)
        self._decoder = ResidualDecoderV2(input_shape[0], embedding_dim, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens, decoder_activation)

    def encode(self, x):
        return self._pre_vq_conv(self._encoder(x))
    
    def quantize(self, x, training):
        quantized = self._vq(x, training)
        return quantized, self._vq.last_codebook_loss, self._vq.last_commitment_loss
    
    def decode(self, z):
        return self._decoder(z)
    
    def forward(self, x, training):
        x = self.encode(x)
        quantized, codebook_loss, commitment_loss = self.quantize(x, training)
        recon = self.decode(quantized)
        return quantized, recon, codebook_loss, commitment_loss
