from typing import Union
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


def combine_losses(*losses_with_weights):
    """
    Combines multiple loss functions into a single callable.

    :param losses_with_weights: List of tuples (loss_fn, weight)
    :return: A callable that computes the weighted sum of all losses
    """
    def _(prediction, target, *args):
        total_loss, all_losses = 0.0, []
        for loss_fn, weight in losses_with_weights:
            loss_value = loss_fn(prediction, target, *args)
            all_losses.append(loss_value)
            total_loss += weight * loss_value
        return total_loss, *all_losses
    return _

def mse_loss(prediction, target, reduction='mean'):
    val = F.mse_loss(prediction, target, reduction=reduction)
    return val

def create_mse_loss(reduction='mean'):
    def _loss(prediction, target, *args):
        return mse_loss(prediction, target, reduction=reduction)
    return _loss

def kl_loss(mu, log_var, reduce=True):
    log_var = torch.clamp(log_var, min=-10, max=10)
    kl_sum = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if reduce:
        return kl_sum.mean()
    return kl_sum

def create_kl_loss(reduce=True):
    def _loss(prediction, target, *args):
        return kl_loss(*args, reduce=reduce)
    return _loss

def create_ssim_loss_function(win_size=7, reduction="elementwise_mean"):
    """
    :win_size: Kernel size to check for similarities
    :param reduction: Type of reduction to apply to the loss tensor
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=win_size, reduction=reduction)
    def _loss(prediction, target, *args):
        return 1 - ssim(prediction, target)
    return _loss

def simple_mse_kl_loss(prediction, target, mu, log_var, beta = 1):
    log_var = torch.clamp(log_var, min=-10, max=10)
    mse_loss = F.mse_loss(prediction, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss *= beta
    return mse_loss + kl_loss, mse_loss, kl_loss
