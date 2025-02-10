import torch
import torch.nn.functional as F


def mse_loss(prediction, target, reduction='mean'):
    val = F.mse_loss(prediction, target, reduction=reduction)
    return val, 


def simple_mse_kl_loss(prediction, target, mu, log_var, beta = 1):
    log_var = torch.clamp(log_var, min=-10, max=10)
    mse_loss = F.mse_loss(prediction, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss *= beta
    return mse_loss + kl_loss, mse_loss, kl_loss