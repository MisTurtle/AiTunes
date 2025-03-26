from abc import ABC, abstractmethod
from typing import Sequence, Type, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ActivationType type hint shortcut
ActivationType = Union[None, Type[nn.Module]]


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
            dummy_encoded = self.encode(torch.rand((1, *self.input_shape)))
            if isinstance(dummy_encoded, tuple):
                dummy_encoded = dummy_encoded[0]
            self._latent_shape = dummy_encoded.shape[1:]
        return self._latent_shape

    @abstractmethod
    def forward(self, x, training=False) -> tuple[torch.Tensor, torch.Tensor, ]:
        pass

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample a batch of n tensors from the latent. These latent samples are not decoded.

        Args:
            n (int): Batch size

        Returns:
            torch.Tensor: A batch of tensors sampled from the latent distribution
        """
        pass

    @abstractmethod
    def decode(self, z):
        pass


###################################
### Vanilla AutoEncoder Classes ###
###################################

class VanillaEncoder(nn.Module):

    def __init__(self, dimensions: Sequence[int]):
        """
        Vanilla Encoder class.

        Args:
            dimensions (Sequence[int]): List of dimensions serving as input and output features to fully-connected Linear layers
        """
        super().__init__()
        self._dimensions = dimensions
        self._layers = nn.Sequential()
        self._build_layers()
    
    def _build_layers(self):
        for i in range(len(self._dimensions) - 1):
            self._layers.append(nn.Linear(self._dimensions[i], self._dimensions[i + 1]))
            self._layers.append(nn.ReLU())
    
    def forward(self, x):
        return self._layers(x)


class VanillaDecoder(nn.Module):

    def __init__(self, dimensions: Sequence[int], activation_type: ActivationType):
        """
        Vanilla Decoder class.

        Args:
            dimensions (Sequence[int]): List of dimensions serving as input and output features to fully-connected Linear layers
            activation_type (ActivationType): Activation function to apply to the decoder output.
        """
        super().__init__()
        self._dimensions = dimensions
        self._layers = nn.Sequential()
        self._activation = activation_type() if activation_type is not None else None
        self._build_layers()
    
    def _build_layers(self):
        for i in range(len(self._dimensions) - 1):
            self._layers.append(nn.Linear(self._dimensions[i], self._dimensions[i + 1]))
            if i != len(self._dimensions) - 2:
                self._layers.append(nn.ReLU())
    
    def forward(self, x):
        x = self._layers(x)
        if self._activation is not None:
            x = self._activation(x)
        return x


class VanillaAutoEncoder(AiTunesAutoencoderModule):

    def __init__(self, input_shape: int, hidden_layer_dimensions: Sequence[int], latent_dimension: int, decoder_activation: ActivationType = None):
        """
        Vanilla Autoencoder Module, using simple fully-connected Linear layers to encoder and decode the input data

        Args:
            input_shape (int): Input data dimension (1d, linear)
            hidden_layer_dimensions (Sequence[int]): Dimensions to map the input data to. More layers means more complexe mathematical operations can be performed
            latent_dimension (int): Final dimension to which the inputs are mapped before being passed to the decoder
            decoder_activation (ActivationType, optional): Activation function to apply to the decoder output. Defaults to None.
        """
        super().__init__((input_shape, ), True)
        self.dimensions = [input_shape] + list(hidden_layer_dimensions) + [latent_dimension]
        self._encoder = VanillaEncoder(dimensions=self.dimensions)
        self._decoder = VanillaDecoder(dimensions=self.dimensions[::-1], activation_type=decoder_activation)

    def encode(self, x):
        return self._encoder(x)

    def decode(self, z):
        return self._decoder(z)
    
    def sample(self, n):
        return torch.randn((n, self.dimensions[-1]))
    
    def forward(self, x, training=False):
        z = self.encode(x)
        return z, self.decode(z)


#######################################
### Variational AutoEncoder classes ###
#######################################

def reparameterize(mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(sigma)
    return mu + sigma * epsilon


class VariationalAutoEncoder(AiTunesAutoencoderModule):

    def __init__(self, input_shape: int, hidden_layers_dimensions: Sequence[int], latent_dimension: int, decoder_activation: ActivationType = None):
        super().__init__((input_shape, ), True)
        self.dimensions = [input_shape] + list(hidden_layers_dimensions) + [latent_dimension]
        self._encoder = VanillaEncoder(dimensions=self.dimensions[:-1])  # Encode up to before the latent dimension
        self._decoder = VanillaDecoder(dimensions=self.dimensions[::-1], activation_type=decoder_activation)  # Decode from the latent dimension
        self._mu = nn.Linear(self.dimensions[-2], self.dimensions[-1])
        self._log_var = nn.Linear(self.dimensions[-2], self.dimensions[-1])
    
    def encode(self, x):
        x = self._encoder(x)
        mu = self._mu(x)
        log_var = self._log_var(x)
        return mu, log_var

    def decode(self, z):
        return self._decoder(z)
    
    def sample(self, n):
        return torch.randn((n, self.dimensions[-1]))

    def forward(self, x, training=False):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var


#################################
### Convolutional VAE classes ###
#################################

def compute_transpose_padding(kernel_size: int, padding: int, stride: Union[int, tuple[int, int]], input_shape: tuple[int, int], output_shape: tuple[int, int]):
    """
    Computes padding and output_padding for ConvTranspose2d layers

    Args:
        kernel_size (int): Kernel size applied to the corresponding Conv2d layer
        padding (int): Padding applied to the corresponding Conv2d layer
        stride (Union[int, tuple[int, int]]): Stride applied to the corresponding Conv2d layer
        input_shape (tuple[int, int]): Input shape passed as argument to the target ConvTranspose2d layer (Height, Width)
        output_shape (tuple[int, int]): Target output shape (Height, Width)

    Returns:
        tuple[int, int]: padding_height, padding_width
    """
    if not isinstance(stride, tuple):
        stride = stride, stride
    h_out = (input_shape[0] - 1) * stride[0] - 2 * padding + 1 * (kernel_size - 1) + 1
    w_out = (input_shape[1] - 1) * stride[1] - 2 * padding + 1 * (kernel_size - 1) + 1
    p_h, p_w = output_shape[0] - h_out, output_shape[1] - w_out

    # --- Debug
    # print("--- ")
    # print("ConvTranspose2d Input Shape:", input_shape)
    # print("ConvTranspose2d Target Output Shape:", output_shape)
    # print("ConvTranspose2d Current Output Shape:", (h_out, w_out)),
    # print("Computed Padding (H, W):", (p_h, p_w))

    return p_h, p_w


class ConvolutionalEncoder(nn.Module):

    def __init__(self, input_shape: Sequence[int], conv_channels: Sequence[int], conv_kernels: Sequence[int], conv_strides: Sequence[int], latent_dimension: int):
        super().__init__()
        self._input_shape = input_shape
        self._channels = conv_channels
        self._kernels = conv_kernels
        self._strides = conv_strides
        self._latent_dimension = latent_dimension

        self._shapes = [input_shape[1:]]  # Get rid of the channels
        self._layers = nn.Sequential()
        self._build_layers()
        self._mu = nn.Linear(self._channels[-1] * np.prod(self._shapes[-1]), self._latent_dimension)
        self._log_var = nn.Linear(self._channels[-1] * np.prod(self._shapes[-1]), self._latent_dimension)

    @property
    def shapes(self) -> list[tuple[int, int]]:
        return self._shapes
    
    def _build_layers(self):
        dummy = torch.randn((1, *self._input_shape))
        for i in range(len(self._channels) - 1):
            self._layers.append(nn.Conv2d(
                in_channels=self._channels[i],
                out_channels=self._channels[i + 1],
                kernel_size=self._kernels[i],
                stride=self._strides[i],
                padding=self._kernels[i] // 2
            ))
            self._layers.append(nn.BatchNorm2d(self._channels[i + 1]))
            self._layers.append(nn.ReLU())
            self._shapes.append(self._layers(dummy).shape[2:])  # Get rid of the batch size and channels (extract H, W)
        self._layers.append(nn.Flatten())
    
    def forward(self, x):
        x = self._layers(x)
        mu = self._mu(x)
        log_var = self._log_var(x)
        return mu, log_var


class ConvolutionalDecoder(nn.Module):
    
    def __init__(self, target_shapes: Sequence[int], conv_channels: Sequence[int], conv_kernels: Sequence[int], conv_strides: Sequence[int], latent_dimension: int, activation_type: ActivationType = None):
        super().__init__()
        self._target_shapes = target_shapes
        self._channels = conv_channels
        self._kernels = conv_kernels
        self._strides = conv_strides
        self._latent_dimension = latent_dimension
        self._activation_type = activation_type

        self._layers = nn.Sequential()
        self._build_layers()
    
    def _build_layers(self):
        self._layers.append(nn.Linear(self._latent_dimension, self._channels[0] * np.prod(self._target_shapes[0])))
        self._layers.append(nn.Unflatten(1, (self._channels[0], *self._target_shapes[0])))
        for i in range(len(self._channels) - 1):
            p_h, p_w = compute_transpose_padding(self._kernels[i], self._kernels[i] // 2, self._strides[i], self._target_shapes[i], self._target_shapes[i + 1])
            self._layers.append(nn.ConvTranspose2d(
                in_channels=self._channels[i],
                out_channels=self._channels[i + 1],
                kernel_size=self._kernels[i],
                stride=self._strides[i],
                padding=self._kernels[i] // 2,
                output_padding=(p_h, p_w)
            ))
            if i != len(self._channels) - 2:
                self._layers.append(nn.BatchNorm2d(self._channels[i + 1]))
                self._layers.append(nn.ReLU())
        
        if self._activation_type is not None:
            self._layers.append(self._activation_type())
    
    def forward(self, x):
        return self._layers(x)

        
class CVAE(AiTunesAutoencoderModule):

    def __init__(self, input_shape, conv_channels: Sequence[int], conv_kernels: Sequence[int], conv_strides: Sequence[int], latent_dimension: int, decoder_activation: ActivationType = None):
        super().__init__(input_shape, False)
        self._conv_channels = [input_shape[0]] + list(conv_channels)
        self._conv_kernels = conv_kernels
        self._conv_strides = conv_strides
        self._latent_dimension = latent_dimension
        
        self._encoder = ConvolutionalEncoder(
            input_shape,
            self._conv_channels,
            self._conv_kernels,
            self._conv_strides,
            self._latent_dimension
        )
        self._decoder = ConvolutionalDecoder(
            self._encoder.shapes[::-1],
            self._conv_channels[::-1],
            self._conv_kernels[::-1],
            self._conv_strides[::-1],
            self._latent_dimension,
            decoder_activation
        )

    def encode(self, x):
        mu, log_var = self._encoder(x)
        return mu, log_var

    def decode(self, z):
        return self._decoder(z)
    
    def sample(self, n):
        return torch.randn((n, self._latent_dimension))
    
    def forward(self, x, training=False):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var


###################################################
### Convolutionnal VAE classes (Smarter ResNet) ###
###################################################
# Inspired from https://github.com/LukeDitria/CNN-VAE/blob/master/RES_VAE.py

class ResnetDownsamplerV1(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, bn_momentum: float = 0.1):
        super().__init__()
        self._channels_in = channels_in
        self._channels_out = channels_out
        self._kernel_size = kernel_size
        
        self._conv1 = nn.Conv2d(channels_in, channels_out // 2, kernel_size, stride=2, padding=kernel_size // 2)
        self._bn1 = nn.BatchNorm2d(channels_out // 2, momentum=bn_momentum)
        self._conv2 = nn.Conv2d(channels_out // 2, channels_out, kernel_size, stride=1, padding=kernel_size // 2)
        self._bn2 = nn.BatchNorm2d(channels_out, momentum=bn_momentum)
        self._conv_skipping = nn.Conv2d(channels_in, channels_out, kernel_size, stride=2, padding=kernel_size // 2)
        self._activation = nn.ReLU()
    
    def forward(self, x):
        skipping = self._conv_skipping(x)
        x = self._activation(self._bn1(self._conv1(x)))
        x = self._activation(self._bn2(self._conv2(x) + skipping))
        return x


class ResnetUpsamplerV1(nn.Module):
    """
    Simple ResNet UpSampler
    """

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, output_padding: tuple[int, int], bn_momentum: float = 0.1):
        super().__init__()
        self._channels_in = channels_in
        self._channels_out = channels_out
        self._kernel_size = kernel_size
        self._output_padding = output_padding

        self._conv1 = nn.ConvTranspose2d(channels_in, channels_in // 2, kernel_size, stride=2, padding=kernel_size // 2, output_padding=self._output_padding)
        self._bn1 = nn.BatchNorm2d(channels_in // 2, momentum=bn_momentum)
        self._conv2 = nn.ConvTranspose2d(channels_in // 2, channels_out, kernel_size, stride=1, padding=kernel_size // 2)
        self._bn2 = nn.BatchNorm2d(channels_out, momentum=bn_momentum)
        self._conv_skipping = nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride=2, padding=kernel_size // 2, output_padding=self._output_padding)
        self._activation = nn.ReLU()
        
    def forward(self, x):
        skipping = self._conv_skipping(x)
        x = self._activation(self._bn1(self._conv1(x)))
        x = self._activation(self._bn2(self._conv2(x) + skipping))
        # x = self._activation(self._bn2(self._conv2(x)) + skipping)  # TODO : Try this out instead?
        return x
        

class ResNet2dV1(AiTunesAutoencoderModule):  # Again, this could reuse the CVAE class by passing some generator for encoder and decoder modules, but would make the structure less readable (and I'm lazy, so yeah)

    def __init__(self, input_shape: Sequence[int], residual_blocks: int, residual_channels: int, latent_space_dim: int, decoder_activation: ActivationType = None, bn_momentum: float = 0.1):
        super().__init__(input_shape=input_shape, flatten=False)
        self._residual_block_count = residual_blocks
        self._residual_channels = residual_channels
        self._latent_space_dim = latent_space_dim
        self._decoder_activation = decoder_activation
        self._bn_momentum = bn_momentum
        # Encoder shapes tracking to be used in the decoder
        self._shapes = [self.input_shape[1:]]
        self._shape_before_bottleneck = None
        # Modules
        self._encoder = self._create_encoder()
        self._decoder = self._create_decoder()        
        self._mu = nn.Linear(np.prod(self._shape_before_bottleneck), self._latent_space_dim)
        self._log_var = nn.Linear(np.prod(self._shape_before_bottleneck), self._latent_space_dim)

    def _create_encoder(self):
        encoder = nn.Sequential()
        encoder.append(nn.Conv2d(self.input_shape[0], self._residual_channels, kernel_size=3, stride=2, padding=1))
        encoder.append(nn.ReLU())
        dummy = torch.tensor(np.random.random((1, *self.input_shape)), dtype=torch.float32)

        channels_in = self._residual_channels
        self._shapes.append(encoder(dummy).shape[2:])
        for _ in range(self._residual_block_count):
            encoder.append(ResnetDownsamplerV1(channels_in, channels_in * 2, 3, self._bn_momentum))
            self._shapes.append(encoder(dummy).shape[2:])
            channels_in *= 2
        self._shape_before_bottleneck = encoder(dummy).shape[1:]
        encoder.append(nn.Flatten())
        return encoder
    
    def _create_decoder(self) -> list[nn.Module]:
        decoder = nn.Sequential()
        decoder.append(nn.Linear(self._latent_space_dim, np.prod(self._shape_before_bottleneck)))
        decoder.append(nn.Unflatten(1, self._shape_before_bottleneck))

        channels_in = self._residual_channels * (2 ** self._residual_block_count)
        for i in range(self._residual_block_count):
            input_shape, output_shape = self._shapes[::-1][i], self._shapes[::-1][i + 1]
            output_padding = compute_transpose_padding(kernel_size=3, padding=3 // 2, stride=2, input_shape=input_shape, output_shape=output_shape)
            decoder.append(ResnetUpsamplerV1(channels_in, channels_in // 2, 3, output_padding, self._bn_momentum))
            channels_in //= 2
        
        output_padding = compute_transpose_padding(kernel_size=3, padding=1, stride=2, input_shape=self._shapes[1], output_shape=self._shapes[0])
        decoder.append(nn.ConvTranspose2d(self._residual_channels, self.input_shape[0], kernel_size=3, stride=2, padding=1, output_padding=output_padding))
        if self._decoder_activation is not None:
            decoder.append(self._decoder_activation())
        return decoder
        
    def encode(self, x):
        x = self._encoder(x)
        return self._mu(x), self._log_var(x)
    
    def decode(self, z):
        return self._decoder(z)
    
    def sample(self, n):
        return torch.randn((n, self._latent_space_dim))
    
    def forward(self, x, training=False):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var
    
    
####################################################################################
### Even smarter 2d residual network, with easy support for deeper architectures ###
####################################################################################
# Performance for the previous ResNet architecture would probably be very similar if support for additional residual layers without downsampling was added 

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
    
    def __init__(self, input_shape, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens):
        """
        Encoder module for a residual network.

        Args:
            input_shape (tuple[int]): Size of the input (Only channels should be required, but we also need to compute the padding for transpose convolutions...)
            num_hiddens (int): Number of convolution channels to reach within the encoder
            num_downsampling_layers (int): Number of downsampling layers to create in the Encoder. Each layer basically divides the input shape by 2. Min: 2 layers
            num_residual_layers (int): Number of residual connections to create in the Encoder and Decoder ResidualStacks
            num_residual_hiddens (int): Number of convolution channels to compress to in the ResidualStacks
        """
        super().__init__()
        self._in_channels = input_shape[0]
        self._input_shape = input_shape
        self._num_hiddens = num_hiddens
        self._num_downsampling_layers = num_downsampling_layers
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._convolution, self._residual_stack = None, None
        self._shapes = [self._input_shape[1:]]
        self._build_modules()

    @property
    def shapes(self):
        return self._shapes
    
    def _build_modules(self):
        layers = nn.Sequential()
        in_channels = self._in_channels 
        dummy = torch.randn(self._input_shape).unsqueeze(0)
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
            self._shapes.append(layers(dummy).shape[2:])
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

    def __init__(self, output_shape, embedding_dim, num_hiddens, num_upsampling_layers, num_residual_layers, num_residual_hiddens, expected_shapes, activation_type: Union[None, nn.Module] = None):
        """
        Decoder module for a residual network.

        Args:
            output_shape (tuple[int]): Expected output size. Only channels should be required, but we also need to know which padding to apply.
            embedding_dim (int): Embedding vectors dimension
            num_embeddings (int): Number of discrete embedding vectors
            num_upsampling_layers (int):  Number of upsampling layers to create. Min: 2 layers
            num_residual_layers (int): Number of residual connections to create in the ResidualStack
            num_residual_hiddens (int): Number of convolution channels to compress to in the ResidualStack
            expected_shapes (tuple[int]): Shapes to match at each step of the decoder through padding
            activation_type (Union[None, nn.Module], optional): Activation function out of the decoder. If None, no activation is applied. Defaults to None. Defaults to None.
        """
        super().__init__()
        self._out_channels = output_shape[0]
        self._output_shape = output_shape
        self._embedding_dim = embedding_dim
        self._num_hiddens = num_hiddens
        self._num_upsampling_layers = num_upsampling_layers
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._activation_type = activation_type
        self._conv_in, self._upconv, self._residual_stack = None, None, None
        self._expected_shapes = expected_shapes
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
            
            output_padding = compute_transpose_padding(4, 1, 2, self._expected_shapes[i], self._expected_shapes[i + 1])
            layers.append(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=output_padding
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
# Inspired from https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
# combined with https://github.com/airalcorn2/vqvae-pytorch/blob/master/vqvae.py
# and https://www.youtube.com/watch?v=1ZHzAOutcnw&ab_channel=ExplainingAI

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
        self.register_buffer('counter', torch.zeros([1], dtype=torch.int64))
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

    def __init__(self, embedding_dim: int, num_embeddings: int, ema: bool = True, random_restart: int = 0, decay: float = 0.99, epsilon: float = 1e-5, restart_threshold: int = 1):
        """
        VectorQuantizer takes an input vector and maps it to the closest in the Embedding Table, performing an Exponential Moving Average iteration if enabled

        Args:
            embedding_dim (int): Embedding vectors dimension
            num_embeddings (int): Number of discrete embedding vectors
            ema (bool, optional): Whether to use Sonnet's Exponential Moving Average to prevent codebook collapse. Defaults to True.
            random_restart (int, optional): How often to perform a random codebook restart, a strategy to fight against codebook collapse Defaults to 0, means never.
            decay (float, optional): Weight given to past values. Closer to one means a smoother and slower reaction to changes. Defaults to 0.99.
            epsilon (float, optional): Epsilon to prevent numerical instability. Defaults to 1e-5.
            restart_threshold (int, optional): Threshold under which codebook entries will be reset after ``restart_every`` batches
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.random_restart, self.restart_in = random_restart, random_restart
        self.ema = ema
        self.restart_threshold = restart_threshold

        # Vector Embedding Table
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embeddings.weight.data.uniform_(-1, 1)  # Make the embeddings close to each other so encoder outputs aren't inherently closer to a few ones
        self.used_embeddings = torch.zeros(self.embeddings.weight.shape[0])

        # EMA var
        self.EMA_cluster_counts = SonnetEMA(decay, (num_embeddings, ))
        self.EMA_embeddings = SonnetEMA(decay, self.embeddings.weight.T.shape)

        # Last loss value (retrieved as properties so it doesn't break torchsummary)
        self.last_codebook_loss = torch.tensor(0.0, requires_grad=True)
        self.last_commitment_loss = torch.tensor(0.0, requires_grad=True)
        self.last_perplexity = torch.tensor(0.0, requires_grad=False)

    def sample(self, n, target_shape):
        indices = torch.randint(0, self.num_embeddings, (n, np.prod(target_shape)))
        quantized = torch.index_select(self.embeddings.weight, 0, indices.view(-1))  # Batch Size x Height x Width, Embedding dim 
        return quantized.reshape((n, self.embedding_dim, *target_shape))

    def forward(self, x, training=False):
        # Reshape the input to isolate the batch size and embedding dimension, getting a list of feature vectors the size of the embedding table
        flat_x = x.permute(0, 2, 3, 1)  # Batch Size, Height, Width, Embedding dim
        flat_x = flat_x.reshape((flat_x.shape[0], -1, flat_x.shape[-1]))  # Batch Size, Height x Width, Embedding dimension 
        
        # Compute distances to each discrete embedding vector 
        lookup_weights = self.embeddings.weight[None, :].repeat((flat_x.shape[0], 1, 1))
        distances = torch.cdist(flat_x, lookup_weights)  # Batch Size, Height x Width, # of discrete vectors  (=> 1 distance per latent "pixel")
        
        # Extract the discrete vectors which are closest to the encoder's output features from the embedding table 
        encoding_indices = distances.argmin(dim=-1)  # Batch Size, Height x Width  (Index to the closest embedding vector for each latent "pixel")
        quantized = torch.index_select(self.embeddings.weight, 0, encoding_indices.view(-1))  # Batch Size x Height x Width, Embedding dim 
        flat_x = flat_x.reshape((-1, flat_x.shape[-1]))  # Batch Size x Height x Width, Embedding dim
        
        # Compute losses to update the encoder so it matches the embedding vectors more closely
        self.last_codebook_loss = None if self.ema else torch.mean((quantized - flat_x.detach()) ** 2)  # Quantized vectors should be close to encoder output
        self.last_commitment_loss = torch.mean((flat_x - quantized.detach()) ** 2)  # Encoder output should be close to quantized vectors

        # Zeros everywhere, except for a 1 at the index of the discrete vector in the embedding table
        encodings = F.one_hot(encoding_indices.view(-1), num_classes=self.num_embeddings).float()  # Batch Size x Height x Width, # of discrete vectors
        
        if self.ema and training:
            # Perform Sonnet Exponential Moving Average updates manually
            with torch.no_grad():
                # Compute usage count for each embedding
                cluster_count = encodings.sum(dim=0)
                updated_ema_cluster_size = self.EMA_cluster_counts(cluster_count)
                
                # Update embeddings EMA
                dw = torch.matmul(flat_x.T, encodings)
                updated_ema_dw = self.EMA_embeddings(dw)
                
                # Update embedding table
                n = updated_ema_cluster_size.sum()
                updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
                normalised_updated_ema_w = updated_ema_dw / updated_ema_cluster_size.unsqueeze(0)
                self.embeddings.weight.copy_(normalised_updated_ema_w.t())
                
                # Random codebook restart
                if self.random_restart > 0:
                    self.used_embeddings += cluster_count
                    self.restart_in -= 1
                    if self.restart_in == 0:
                        self.restart_in = self.random_restart
                        dead_indices = torch.where(self.used_embeddings < self.restart_threshold)[0]
                        if dead_indices.numel() > 0:
                            # Reset dead entries with samples from the current batch
                            rand_indices = torch.randint(0, flat_x.size(0), (dead_indices.shape[0], ))
                            new_entries = flat_x[rand_indices]
                            self.embeddings.weight[dead_indices] = new_entries
                        self.used_embeddings = torch.zero_(self.used_embeddings)
    
        # Straight Through Estimation (To keep gradient flow going even with the undifferentiable table lookup)
        quantized = flat_x + (quantized - flat_x).detach()
        quantized = quantized.reshape(x.shape)  # Batch Size, Channels, Height, Width
        
        # Perplexity computation (codebook usage)
        avg_probs = torch.mean(encodings, 0)
        self.last_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

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
            random_restart: int = 0,
            decay: float = 0.99,
            epsilon: float = 1e-5,
            restart_threshold: int = 1
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
            random_restart (int, optional): How often to perform a random codebook restart, a strategy to fight against codebook collapse Defaults to 0, means never.
            decay (float, optional): Weight given to past values. Closer to one means a smoother and slower reaction to changes. Defaults to 0.99.
            epsilon (float, optional): Epsilon to prevent numerical instability. Defaults to 1e-5.
        """
        super().__init__(input_shape=input_shape, flatten=False)

        self._encoder = ResidualEncoderV2(input_shape, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1)
        self._vq = VectorQuantizer(embedding_dim, num_embeddings, use_ema, random_restart, decay, epsilon, restart_threshold)
        self._decoder = ResidualDecoderV2(input_shape, embedding_dim, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens, self._encoder.shapes[::-1], decoder_activation)

    def encode(self, x):
        return self._pre_vq_conv(self._encoder(x))
    
    def quantize(self, x, training):
        quantized = self._vq(x, training)
        return quantized, self._vq.last_codebook_loss, self._vq.last_commitment_loss, self._vq.last_perplexity
    
    def decode(self, z):
        return self._decoder(z)
    
    def sample(self, n):
        return self._vq.sample(n, self._encoder.shapes[-1])
    
    def forward(self, x, training=False):
        x = self.encode(x)
        quantized, codebook_loss, commitment_loss, perplexity = self.quantize(x, training)
        recon = self.decode(quantized)
        return quantized, recon, codebook_loss, commitment_loss, perplexity
