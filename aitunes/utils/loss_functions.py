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
        components = []
        for loss_fn, weight in losses_with_weights:
            loss_value = loss_fn(prediction, target, *args)[0]
            if loss_value is None:
                continue
            components.append(loss_value * weight)
        return torch.stack(components).sum(), *components
    return _

def mse_loss(prediction, target, reduction='mean'):
    val = F.mse_loss(prediction, target, reduction=reduction)
    return val,

def create_mse_loss(reduction='mean'):
    def _loss(prediction, target, *args):
        return F.mse_loss(prediction, target, reduction=reduction),
    return _loss

def kl_loss(mu, log_var, reduce=True):
    log_var = torch.clamp(log_var, min=-10, max=10)
    kl_sum = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if reduce:
        return kl_sum / mu.size(0),
    return kl_sum,

def create_kl_loss(reduce=True):
    def _loss(prediction, target, *args):
        return kl_loss(*args, reduce=reduce)
    return _loss

def create_kl_loss_with_linear_annealing(over_epochs: int, batch_per_epoch: int, start_weight: float = 0, end_weight: float = 1, reduce=True):
    _loss_weight = start_weight
    _batches = 0
    def _loss(prediction, target, *args):
        nonlocal _batches, _loss_weight
        _batches += 1
        _progress = _batches / (over_epochs * batch_per_epoch)
        _loss_weight = min(end_weight, start_weight + (end_weight - start_weight) * _progress)
        kl = kl_loss(*args, reduce=reduce)
        return _loss_weight * kl[0], 
    return _loss

def create_kl_loss_with_cyclic_annealing(over_epochs: int, batch_per_epoch: int, start_weight: float = 0, end_weight: float = 1, reduce=True):
    _loss_weight = start_weight
    _batches = 0
    def _loss(prediction, target, *args):
        nonlocal _batches, _loss_weight
        _batches += 1
        _progress = (_batches / (over_epochs * batch_per_epoch)) % 1
        _loss_weight = start_weight + (end_weight - start_weight) * _progress
        kl = kl_loss(*args, reduce=reduce)
        return _loss_weight * kl[0], 
    return _loss

def create_ssim_loss_function(win_size=7, reduction="elementwise_mean"):
    """
    :win_size: Kernel size to check for similarities
    :param reduction: Type of reduction to apply to the loss tensor
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=win_size, reduction=reduction)
    def _loss(prediction, target, *args):
        return 1 - ssim(prediction, target),
    return _loss

def create_cherry_picked_loss(indices: tuple[int], weights: tuple[float]):
    """
    > Created for VQ VAE models, to directly retrieve loss components returned by the model
    :param indices: *args indices to the raw losses returned by the model. args[0] is the first argument after the returned reconstruction value
    :param weights: Weight for each raw loss to be added together
    """
    def _loss(prediction, target, *args):
        loss = torch.tensor(0.0, requires_grad=True)
        for weight_i, index in enumerate(indices):
            if args[index] is None:
                continue
            loss = loss + args[index] * weights[weight_i]
        return loss, 
    return _loss


def simple_mse_kl_loss(prediction, target, mu, log_var, beta = 1):
    log_var = torch.clamp(log_var, min=-10, max=10)
    mse_loss = F.mse_loss(prediction, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss *= beta
    return mse_loss + kl_loss, mse_loss, kl_loss
