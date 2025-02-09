from os import path, listdir, makedirs
from typing import Any, Callable, Union
from abc import ABC, abstractmethod
from datetime import datetime
from aitunes.utils import get_loading_char
from aitunes.modules import SimpleAutoEncoder as SAE, VariationalAutoEncoder as VAE

import time
import numpy as np
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

class AutoencoderExperiment(ABC):
    """
    Class describing a test case for an AE, VAE or CVAE model.
    Anything that is not directly related to the test contents or the model is handled by an AutoencoderExperimentSupport instance for better visibility (maybe)
    """

    def __init__(self, name: str, model: Union[SAE, VAE], weights_path: str, loss_criterion: nn.Module, optimizer: optim.Optimizer):
        self._model = model
        self._weights_path = weights_path
        self._support = AutoencoderExperimentSupport(name)

        self._loss_criterion: nn.Module = loss_criterion
        self._optimizer: optim.Optimizer = optimizer
        self._middlewares = []  # Middlewares applied when evaluating the model (Probably for plotting visual information)

        self._load_weights()

    @property
    def model(self):
        return self._model
    
    def set_plotting(self, plotting: bool) -> 'AutoencoderExperiment':
        self._support.plotting = plotting
    
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
            checkpoint = torch.load(self._weights_path, weights_only=True)
            self._model.load_state_dict(checkpoint["model"])
            self._optimizer.load_state_dict(checkpoint["optimizer"])

            self._support.ran_epochs = checkpoint["epoch"]
            self._support.total_epochs += self._support.ran_epochs

            self._support.trained = True
            self._support.log(f"Loaded weights from file {self._weights_path}")
        else:
            self._support.log(f"Weights could not be found at path {self._weights_path}")
    
    def _save_weights(self, to: Union[None, str] = None):
        save_path = to or self._weights_path
        makedirs(path.dirname(save_path), exist_ok=True)
        torch.save({
            "epoch": self._support.ran_epochs,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict()
        }, save_path)
        self._support.log(f"Saved model weights at path {save_path}")
    
    def save_when(self, check: Callable[[int, float], bool], root_folder: str):
        """
        :param check: A callable taking the current epoch as well as the current loss as a parameter, and returns true if the model should be save
        :param root_folder: A formattable string that points to a folder where to save the model and training data at.
        """
        self._support.save_when(check, root_folder)
    
    def save_every(self, epoch_count: int, root_folder: str):
        """
        A shorthand function to save the model every x epochs
        """
        if epoch_count <= 0:  # If not epoch count is specified, only save an history entry when the training finishes
            self.save_when(lambda *_: False, root_folder)
        else:
            self.save_when(lambda epoch, _: epoch % epoch_count == 0, root_folder)

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
                self._support.add_batch_result(batch_loss.item() / batch.shape[0]).log_training_loss()

            self._support.log_training_loss(ended=True).next_epoch()
            # Save the model only if necessary (checks are performed in the support directly)
            self._support.perform_save(self._save_weights)
        
        self._support.trained = True

    def evaluate(self):
        self._support.log("An evaluation session of the model is about to start...")
        self._support.blank_epoch()
        self._model.eval()
        
        with torch.no_grad():            
            for batch, *extra in self.next_batch(training=False):
                embedding, prediction, *args = self._model(batch)
                batch_loss: torch.Tensor = self._loss_criterion(prediction, batch, *args)
                self.apply_middlewares(batch, prediction, embedding, extra, args)
                self._support.add_batch_result(batch_loss / batch.shape[0]).log_running_loss("Evaluation", False, True)
            self._support.log_running_loss("Evaluation", True, False)
            
    @abstractmethod
    def interactive_evaluation(self):
        """
        Starts an interactive interface where the user can interact and see the live results
        /!\\ Do not forget to disable training mode (For performance) /!\\
        
        TODO : See to use a decorator to remove boilerplate for individual tasks?
        """
        pass


class AutoencoderExperimentSupport:

    def __init__(self, name):
        self._name = name
        self.trained, self.plotting = False, False
        self.total_epochs, self.ran_epochs = 0, 0
        self.all_epoch_losses, self.current_epoch_losses = [], []
        self.current_epoch_loss = .0
        self.epoch_start, self.all_epoch_running_time = time.time(), []
        self.ax, self.fig, self.plot_calls = None, None, 0
        self.save_check, self.save_path = None, None
        self.batch_per_training_epoch = 0
    
    @property
    def prefix(self):
        return f"[{self._name}]"

    @property
    def epoch_loss(self):
        """
        Average loss per item during this epoch
        """
        return sum(self.current_epoch_losses)

    @property
    def epoch_batches(self):
        return len(self.current_epoch_losses)

    @property
    def epoch_runtime(self):
        return time.time() - self.epoch_start
    
    def set_batch_size(self, batch_size: int) -> 'AutoencoderExperimentSupport':
        self.batch_per_training_epoch = batch_size
        return self
    
    def blank_epoch(self) -> 'AutoencoderExperimentSupport':
        self.current_epoch_losses.clear()
        self.epoch_start = time.time()
        return self
    
    def next_epoch(self) -> 'AutoencoderExperimentSupport':
        self.ran_epochs += 1
        self.all_epoch_losses.append(self.epoch_loss)
        self.all_epoch_running_time.append(self.epoch_runtime)
        self.plot_loss_progress(1)
        if not self.trained:
            self.set_batch_size(self.epoch_batches)
        return self.blank_epoch()
    
    def add_total_epochs(self, n: int) -> 'AutoencoderExperimentSupport':
        self.total_epochs += n
        return self
    
    def add_batch_result(self, loss: float) -> 'AutoencoderExperimentSupport':
        """
        Add the average loss per item to the current epoch history
        """
        self.current_epoch_losses.append(loss)
        return self
    
    # SAVING FUNCTIONS BELOW
    def save_when(self, check: Callable[[int, float], bool], root_folder: str):
        self.save_check = check
        self.save_path = root_folder
    
    def perform_save(self, save_to_path_fn: Callable[[str], None]):
        """
        :param save_to_path_fn: A function taking a path as a parameter and saving weights of the model there
        """
        should_save = self.save_check is not None and self.save_check(self.ran_epochs, self.epoch_loss)
        force_save = self.ran_epochs == self.total_epochs
        should_save |= force_save  # Always save if the training is about to end
        
        if not should_save:
            return
        
        if self.save_path is None:  # No history path is provided, save to the main model location
            save_to_path_fn(None)
            return
        
        # Compute the target directory name in the history folder
        dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = path.join(self.save_path, dir_name)

        # Save the model as a checkpoint
        makedirs(dir_path, exist_ok=True)
        save_to_path_fn(path.join(dir_path, f"checkpoint_{self.ran_epochs}.pth" if not force_save else f"release_{self.ran_epochs}.pth"))

        # Save the training progress
        epoch_ids = ['Epoch #', *np.arange(self.ran_epochs - len(self.all_epoch_losses), self.ran_epochs)]
        epoch_losses = ['Total Loss', *self.all_epoch_losses]
        if self.batch_per_training_epoch is not None and self.batch_per_training_epoch > 0:
            epoch_avg_losses = ['Avg Loss', *map(lambda x: x / self.batch_per_training_epoch, self.all_epoch_losses)]
        else:
            epoch_avg_losses = epoch_losses
        epoch_duration = ['Runtime', *self.all_epoch_running_time]

        np.savetxt(
            path.join(dir_path, "progress.csv"),
            [row for row in zip(epoch_ids, epoch_losses, epoch_avg_losses, epoch_duration)],
            delimiter=",",
            fmt="%s"
        )

        # If the training reached the end, save the last model a second time as a release one.
        if force_save:
            save_to_path_fn(None)
        self.epoch_start = time.time()

    
    # PLOTTING FUNCTIONS BELOW
    def plot_loss_progress(self, every: int = 10) -> 'AutoencoderExperimentSupport':
        """
        :param every: Only perform the plotting operation every x calls
        """
        if not self.plotting:
            return self

        self.plot_calls += 1
        if self.plot_calls % every != 0:
            return self
        
        if self.ax is None:
            self.fig, self.ax = plt.subplots()

        plt.ion()
        self.ax.clear()
        self.ax.plot(self.all_epoch_losses, label="Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title(f"{self._name} Training Loss (Live Plot)")
        self.ax.legend()
        plt.draw()
        plt.pause(0.05)
        plt.ioff()
        return self

    # LOGGING FUNCTIONS BELOW

    def log(self, msg: str) -> 'AutoencoderExperimentSupport':
        print(f"{self.prefix} {msg}")
        return self
    
    def get_running_loss_line(self, prefix: str):
        el = self.epoch_loss
        return f"{self.prefix} [{prefix}] Running time: {self.epoch_runtime:.2f}s... Epoch loss: {el:.2f}... Avg. Loss per item: {el / max(1, self.epoch_batches):.4f}"

    def log_running_loss(self, prefix: str, new_line: bool = False, loading: bool = True) -> 'AutoencoderExperimentSupport':
        line = self.get_running_loss_line(prefix)
        if loading:
            line = get_loading_char() + " " + line
        if new_line:
            line += "\n"
        print("\r" + line, end='')
        return self
    
    def log_training_loss(self, ended: bool = False) -> 'AutoencoderExperimentSupport':
        self.log_running_loss(f"Epoch {self.ran_epochs + 1}/{self.total_epochs}", ended, not ended)
        return self