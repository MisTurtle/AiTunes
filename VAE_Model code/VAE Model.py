import torch
import torch.nn as nn
import os
import pickle

#based on https://www.youtube.com/watch?v=A6mdOEPGM1E&ab_channel=ValerioVelardo-TheSoundofAI

class VAE(nn.Module):

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape  
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self._shape_before_bottleneck = self._calculate_shape_before_bottleneck()

        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def forward(self, x):
        mu, log_var = self._encode(x)
        z = self._sample(mu, log_var)
        x_reconstructed = self._decode(z)
        return x_reconstructed, mu, log_var

    def _build_encoder(self):
        layers = []
        input_channels = self.input_shape[0]
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            input_channels = out_channels
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self._shape_before_bottleneck[0], self.latent_space_dim * 2))  # mu and log_var
        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = [nn.Linear(self.latent_space_dim, self._shape_before_bottleneck[0])]
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

    def _calculate_shape_before_bottleneck(self):
        """
        Calcule les dimensions exactes avant la couche de goulot d'étranglement.
        Retourne la taille aplatie totale et les dimensions en tuple.
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

    def compute_loss(self, x, x_reconstructed, mu, log_var):
        reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kl_loss


