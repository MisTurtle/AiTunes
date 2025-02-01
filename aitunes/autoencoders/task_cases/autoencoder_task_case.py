from os import path
from typing import Any, Callable, Union
from abc import ABC, abstractmethod
from aitunes.utils import get_loading_char
from aitunes.autoencoders.autoencoders_modules import SimpleAutoEncoder as SAE, VariationalAutoEncoder as VAE

import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# Function taking 5 parameters:
# - Original batch of data
# - Prediction batch made by the model
# - Latent space embedding
# - A list of extra dataset parameters (like labels and such)
# - A list of extra results yielded by the model (*args)
Middleware = Callable[[torch.tensor, torch.tensor, torch.tensor, list[Any], list[torch.tensor]], None]

FLAG_NONE       = 0b0000  # Clean slate
FLAG_TRAINED    = 0b0001  # On if the model has been trained
FLAG_PLOTTING   = 0b0010  # On if plotting is enabled

class AutoencoderTaskSupport:

    def __init__(self, name, flags=FLAG_NONE):
        self._name = name
        self._state = flags
        self.total_epochs, self.ran_epochs = 0, 0
        self.all_epoch_losses, self.current_epoch_losses = [], []
        self.current_epoch_loss = .0
        self.epoch_start = time.time()
        self.ax, self.fig, self.plot_calls = None, None, 0
    
    @property
    def prefix(self):
        return f"[{self._name}]"

    @property
    def epoch_loss(self):
        return sum(self.current_epoch_losses)

    @property
    def epoch_batches(self):
        return len(self.current_epoch_losses)

    @property
    def epoch_runtime(self):
        return time.time() - self.epoch_start

    def has(self, flag: int) -> bool:
        return bool(self._state & flag)
    
    def set(self, flag: int, val: bool = True) -> 'AutoencoderTaskSupport':
        if val:
            self._state |= flag
        else:
            self.unset(flag)
        return self
    
    def unset(self, flag: int) -> 'AutoencoderTaskSupport':
        self._state &= ~flag
        return self
    
    def blank_epoch(self) -> 'AutoencoderTaskSupport':
        self.current_epoch_losses.clear()
        self.epoch_start = time.time()
        return self
    
    def next_epoch(self) -> 'AutoencoderTaskSupport':
        self.ran_epochs += 1
        self.all_epoch_losses.append(self.epoch_loss)
        self.plot_loss_progress(1)
        return self.blank_epoch()
    
    def add_total_epochs(self, n: int) -> 'AutoencoderTaskSupport':
        self.total_epochs += n
        return self
    
    def add_batch_result(self, loss: float) -> 'AutoencoderTaskSupport':
        self.current_epoch_losses.append(loss)
        return self
    
    # PLOTTING FUNCTIONS BELOW
    def plot_loss_progress(self, every: int = 10) -> 'AutoencoderTaskSupport':
        """
        :param every: Only perform the plotting operation every x calls
        """
        if not self.has(FLAG_PLOTTING):
            return self

        self.plot_calls += 1
        if self.plot_calls % every != 0:
            return self
        
        if self.ax is None:
            # plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.plot(self.all_epoch_losses, label="Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title(f"{self._name} Training Loss (Live Plot)")
        self.ax.legend()
        plt.draw()
        plt.pause(0.1)
        return self

    # LOGGING FUNCTIONS BELOW, CAN BE IGNORED

    def log(self, msg: str) -> 'AutoencoderTaskSupport':
        print(f"{self.prefix} {msg}")
        return self
    
    def get_running_loss_line(self, prefix: str):
        el = self.epoch_loss
        return f"{self.prefix} [{prefix}] Running time: {self.epoch_runtime:.2f}s... Epoch loss: {el:.2f}... Avg. Loss: {el / max(1, self.epoch_batches):.4f}"

    def log_running_loss(self, prefix: str, new_line: bool = False, loading: bool = True) -> 'AutoencoderTaskSupport':
        line = self.get_running_loss_line(prefix)
        if loading:
            line = get_loading_char() + " " + line
        if new_line:
            line += "\n"
        print("\r" + line, end='')
        return self
    
    def log_training_loss(self, ended: bool = False) -> 'AutoencoderTaskSupport':
        self.log_running_loss(f"Epoch {self.ran_epochs + 1}/{self.total_epochs}", ended, not ended)
        return self


class AutoencoderTaskCase(ABC):
    """
    Class describing a test case for an Autoencoder (or VAE) model.
    Anything that is not directly related to the test contents or the model is handled by an AutoencoderTestSupport instance
    """

    def __init__(self, name: str, model: Union[SAE, VAE], weights_path: str, loss_criterion: nn.Module, optimizer: optim.Optimizer, flags: int = FLAG_NONE):
        self._model = model
        self._weights_path = weights_path
        self._support = AutoencoderTaskSupport(name, flags)

        self._loss_criterion: nn.Module = loss_criterion
        self._optimizer: optim.Optimizer = optimizer
        self._middlewares = []  # Middlewares applied when evaluating the model (Probably for plotting visual information)

        self._load_weights()

    @property
    def model(self):
        return self._model
    
    @property
    def trained(self):
        return self._support.has(FLAG_TRAINED)
    
    @property
    def plotting(self):
        return self._support.has(FLAG_PLOTTING)
    
    def add_middleware(self, mw: Middleware):
        self._middlewares.append(mw)
    
    def apply_middlewares(self, original_batch, predicted_batch, embedded_batch, batch_extra, model_extra):
        for mw in self._middlewares:
            mw(original_batch, predicted_batch, embedded_batch, batch_extra, model_extra)

    @abstractmethod
    def next_batch(self, training):
        """
        Generator yielding batches of data to be trained or evaluated on
        It should yield as many batches necessary to complete a full epoch as a tuple (data, extra data like a label)
        """
        yield None, None

    def _load_weights(self):
        if path.exists(self._weights_path):
            self._model.load_state_dict(torch.load(self._weights_path, weights_only=True))
            self._support.set(FLAG_TRAINED)
            self._support.log(f"Loaded weights from file {self._weights_path}")
        else:
            self._support.log(f"Weights could not be found at path {self._weights_path}")
    
    def _save_weights(self):
        torch.save(self._model.state_dict(), self._weights_path)
        self._support.log(f"Saved model weights at path {self._weights_path}")

    def train(self, epochs: int):
        self._model.train(True)
        self._support.add_total_epochs(epochs)
        self._support.log(f"A training session of {epochs} epochs is about to start...")

        for _ in range(epochs):
            for batch, *_ in self.next_batch(training=True):
                self._optimizer.zero_grad()
                # Predict with the current model state and compute the loss
                embedding, prediction, *args = self._model(batch)
                batch_loss: torch.Tensor = self._loss_criterion(prediction, batch, *args)
                batch_loss.backward()
                # Run an optimizer step and log the result
                self._optimizer.step()
                self._support.add_batch_result(batch_loss.item()).log_training_loss()

            self._support.log_training_loss(ended=True).next_epoch()
        # Save the model and mark as trained
        self._save_weights()
        self._support.set(FLAG_TRAINED)

    def evaluate(self):
        self._support.log("An evaluation session of the model is about to start...")
        self._support.blank_epoch()
        self._model.eval()
        
        with torch.no_grad():            
            for batch, *extra in self.next_batch(training=False):
                embedding, prediction, *args = self._model(batch)
                batch_loss: torch.Tensor = self._loss_criterion(prediction, batch, *args)
                self.apply_middlewares(batch, prediction, embedding, extra, args)
                self._support.add_batch_result(batch_loss).log_running_loss("Evaluation", False, True)
            self._support.log_running_loss("Evaluation", True, False)
            
    @abstractmethod
    def interactive_evaluation(self):
        """
        Starts an interactive interface where the user can interact and see the live results
        /!\\ Do not forget to disable training mode (For performance) /!\\
        
        TODO : See to use a decorator to remove boilerplate for individual tasks?
        """
        pass
