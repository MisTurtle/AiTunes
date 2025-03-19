import math
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from os import path, makedirs
from typing import Any, Callable, Iterable, Union
from abc import ABC, abstractmethod
from datetime import datetime

import aitunes.utils as utils
from aitunes.modules import AiTunesAutoencoderModule
from aitunes.utils import get_loading_char, plot_umap
from aitunes.utils.audio_utils import AudioFeatures, audio_model_interactive_evaluation



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

    def __init__(self, name: str, model: AiTunesAutoencoderModule, weights_path: str, loss_criterion: nn.Module, optimizer: optim.Optimizer):
        self._model = model
        self._weights_path = weights_path
        self._support = AutoencoderExperimentSupport(name)

        self._loss_criterion: nn.Module = loss_criterion
        self._optimizer: optim.Optimizer = optimizer
        self._middlewares = []  # Middlewares applied when evaluating the model (Probably for plotting visual information)

        summary(model, model.input_shape, device=utils.device.type)
        self._load_weights()

    @property
    def model(self):
        return self._model
    
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        :return: How many items are in a batch
        """
        pass

    @property
    @abstractmethod
    def batch_per_epoch(self) -> int:
        """
        :return: How many batches are in a training epoch (approximately, mostly used for loss annealing)
        """
        # Bad design choice, sadge, I can't access this for linear annealing from scenarios as the task isn't instantiated yet
        # I also can't have it static or it would make it impossible for spectrogram based experiments to override the value
        pass
    
    def set_plotting(self, plotting: bool) -> 'AutoencoderExperiment':
        self._support.plotting = plotting
    
    def add_middleware(self, mw: Middleware):
        self._middlewares.append(mw)
    
    def apply_middlewares(self, original_batch, predicted_batch, embedded_batch, batch_extra, model_extra):
        for mw in self._middlewares:
            mw(original_batch, predicted_batch, embedded_batch, batch_extra, model_extra)

    @abstractmethod
    def next_batch(self, training, lookup_labels: bool = False):
        """
        Generator yielding batches of data to be trained or evaluated on

        Args:
            training (bool): Which dataset to use
            lookup_labels (bool, optional): Lookup correct labels, maybe slowing down the process. Defaults to False.

        Yields:
            torch.Tensor: as many batches as necessary to complete a full epoch as a tuple (data, extra data like a label)
        """
        yield None, None

    def _load_weights(self):
        if path.exists(self._weights_path):
            checkpoint = torch.load(self._weights_path, weights_only=True, map_location=utils.device)
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
    
    def display_umap_projection(self, on_training: bool = True):
        mu, labels = [], []
        with torch.no_grad():
            for batch, *extra in self.next_batch(on_training, lookup_labels=True):
                embedding, prediction, *args = self.model(batch, training=False)
                if len(extra) > 0 and len(args) > 0:  # First check a label is indeed given by the next_batch function
                    batch_labels = extra[0]
                    mu.append(embedding.cpu().numpy())  # Args[0] is mu
                    if isinstance(batch_labels, torch.Tensor):
                        batch_labels = batch_labels.cpu().numpy()
                    labels.append(batch_labels)
            if len(mu) > 0:
                mu = np.concatenate(mu, axis=0)
                labels = np.concatenate(labels, axis=0)
                print(f"[{self._support._name}] Computing UMAP Projection...")
                plot_umap(mu, labels)

    def train(self, epochs: int):
        self._model.train(True)
        self._support.add_total_epochs(epochs)
        self._support.log(f"A training session of {epochs} epochs is about to start...")

        for _ in range(epochs):
            for batch, *extra in self.next_batch(training=True, lookup_labels=False):
                self._optimizer.zero_grad()
                # Predict with the current model state and compute the loss
                embedding, prediction, *args = self._model(batch, training=True)
                combined_loss, *loss_components = self._loss_criterion(prediction, batch, *args)
                combined_loss.backward()

                # Run an optimizer step and log the result
                self._optimizer.step()
                self._support.add_batch_result(combined_loss.item(), *loss_components).log_training_loss()

            self._support.log_training_loss(ended=True).next_epoch()
            # Save the model only if necessary (checks are performed in the support directly)
            self._support.perform_save(self._save_weights)
        
        self._support.trained = True

    def evaluate(self):
        self._support.log("An evaluation session of the model is about to start...")
        self._support.blank_epoch()
        self._model.eval()
        
        with torch.no_grad():            
            for batch, *extra in self.next_batch(training=False, lookup_labels=False):
                embedding, prediction, *args = self._model(batch, training=False)
                batch_loss, *loss_components = self._loss_criterion(prediction, batch, *args)
                self.apply_middlewares(batch, prediction, embedding, extra, args)
                self._support.add_batch_result(batch_loss, *loss_components).log_running_loss("Evaluation", False, True)
            self._support.log_running_loss("Evaluation", True, False)

    @abstractmethod
    def interactive_evaluation(self):
        """
        Starts an interactive interface where the user can interact and see the live results
        /!\\ Do not forget to disable training mode (For performance) /!\\
        """
        pass


class AutoencoderExperimentSupport:

    def __init__(self, name):
        self._name = name  # Name of the experiment
        self.trained, self.plotting = False, False  # State and settings
        self.total_epochs, self.ran_epochs = 0, 0  # Total of epochs to be performed, total epochs lived by the model

        # History of avg loss per item for the current epoch. List of tuples where the first value is the total loss, and following values are components making up that loss
        self.epoch_mean_item_losses = []
        # History of loss per epoch for all previous epochs. List of tuples where the first value is the total loss, and following values are components making up that loss
        self.all_epoch_item_losses = []
        
        self.epoch_start, self.all_epoch_running_times = time.time(), []
        
        self.ax, self.fig, self.plot_calls = None, None, 0
        self.save_check, self.save_path = None, None
        
    @property
    def prefix(self):
        return f"[{self._name}]"
    
    @property
    def current_average_item_loss(self) -> tuple[float, ...]:
        if len(self.epoch_mean_item_losses) == 0:
            return 0
        return np.mean(self.epoch_mean_item_losses, axis=0)

    @property
    def current_epoch_runtime(self):
        return time.time() - self.epoch_start
    
    def blank_epoch(self) -> 'AutoencoderExperimentSupport':
        self.epoch_mean_item_losses.clear()
        self.epoch_start = time.time()
        return self
    
    def next_epoch(self) -> 'AutoencoderExperimentSupport':
        self.ran_epochs += 1
        self.all_epoch_item_losses.append(self.current_average_item_loss)
        self.all_epoch_running_times.append(self.current_epoch_runtime)
        self.plot_loss_progress(1)
        return self.blank_epoch()
    
    def add_total_epochs(self, n: int) -> 'AutoencoderExperimentSupport':
        self.total_epochs += n
        return self
    
    def add_batch_result(self, loss: float, *loss_components: torch.Tensor) -> 'AutoencoderExperimentSupport':
        """
        Adds a batch loss (avg loss per item) to the current epoch
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if len(loss_components) > 0:
            loss_components = map(lambda lc: lc.detach().cpu().numpy(), loss_components)
        self.epoch_mean_item_losses.append((loss, *loss_components))
        return self
    
    # SAVING FUNCTIONS BELOW
    def save_when(self, check: Callable[[int, float], bool], root_folder: str):
        self.save_check = check
        self.save_path = root_folder
    
    def perform_save(self, save_to_path_fn: Callable[[str], None]):
        """
        :param save_to_path_fn: A function taking a path as a parameter and saving weights of the model there
        """
        should_save = self.save_check is not None and self.save_check(self.ran_epochs, self.current_average_item_loss)
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
        all_epoch_item_losses = np.array(self.all_epoch_item_losses)
        epoch_ids = ['Epoch #', *np.arange(self.ran_epochs - len(self.all_epoch_item_losses), self.ran_epochs)]
        epoch_durations = ['Runtime', *self.all_epoch_running_times]
        epoch_avg_item_losses = ['Item Loss', *all_epoch_item_losses[:, 0]]
        epoch_avg_item_loss_components = [
            [f"Component #{i + 1}"] + all_epoch_item_losses[:, i + 1].tolist()
            for i in range(all_epoch_item_losses.shape[1] - 1)
        ]

        np.savetxt(
            path.join(dir_path, "progress.csv"),
            [row for row in zip(epoch_ids, epoch_durations, epoch_avg_item_losses, *epoch_avg_item_loss_components)],
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
        losses = np.array(self.all_epoch_item_losses)
        self.ax.plot(losses[:, 0], label="Avg Item Loss")
        for i in range(losses.shape[1] - 1):
            self.ax.plot(losses[:, i + 1], label=f"Component #{i + 1}")
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
        # Direct logging is still kept to keep some feedback
        print(f"{self.prefix} {msg}")
        return self
    
    def get_running_loss_line(self, prefix: str):
        if utils.quiet:
            return
        
        # epoch_loss = self.current_epoch_loss
        item_loss = self.current_average_item_loss
        # epoch_loss_str = f"Epoch loss: {epoch_loss[0]:.2f}..."
        item_loss_str = f"Item Loss: {item_loss[0]:.4f}"
        for component_id in range(item_loss.shape[0] - 1):
            item_loss_str += f"... #{component_id}: {item_loss[component_id + 1]:.4f}"
        return f"{self.prefix} [{prefix}] Ran: {self.current_epoch_runtime:.2f}s... {item_loss_str}"

    def log_running_loss(self, prefix: str, new_line: bool = False, loading: bool = True) -> 'AutoencoderExperimentSupport':
        if utils.quiet:
            return self
        
        line = self.get_running_loss_line(prefix)
        if loading:
            line = get_loading_char() + " " + line
        if new_line:
            line += "\n"
        print("\r" + line, end='')
        return self
    
    def log_training_loss(self, ended: bool = False) -> 'AutoencoderExperimentSupport':
        if not utils.quiet:
            self.log_running_loss(f"Epoch {self.ran_epochs + 1}/{self.total_epochs}", ended, not ended)
        return self
    

class SpectrogramBasedAutoencoderExperiment(AutoencoderExperiment):
    # TODO : Move this to Experiments/Cases, it has nothing to do here
    def __init__(self, name, model, weights_path, loss_criterion, optimizer, training_data: h5py.File, evaluation_data: h5py.File, mode: AudioFeatures, batch_size: int, file_name_to_label: Callable = None):
        """
        Autoencoder Experiment based on spectrogram reconstruction

        Args:
            name (str): Experimentation name
            model (nn.Module): Model to train or evaluate
            weights_path (str): Path to the model + optimizer weights
            loss_criterion (Callable): Loss compute method
            optimizer (nn.optim.Optimizer): Optimizer used by the model
            training_data (h5py.File): Training dataset file instance
            evaluation_data (h5py.File): Evaluation dataset file instance
            mode (AudioFeatures): Audio quality settings used during interactive evaluation 
            batch_size (int): # of items per batch
            file_name_to_label (Callable[[str | Iterable[str]], Iterable[str | int]]): Callable to map the original file name to a label
        """
        super().__init__(name, model, weights_path, loss_criterion, optimizer)

        self.train_loader, self.train_labels = training_data["spectrograms"], training_data["labels"]
        self.training_indices = np.arange(len(self.train_loader))
        
        self.test_loader, self.test_labels = evaluation_data["spectrograms"], evaluation_data["labels"]
        self.test_indices = np.arange(len(self.test_loader))

        self.mode = mode
        self._batch_size = batch_size
        self._file_name_to_label = file_name_to_label
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def batch_per_epoch(self) -> int:
        return int(math.ceil(len(self.train_loader) / self.batch_size))

    def next_batch(self, training: bool, lookup_labels: bool = False):
        dataset = self.train_loader if training else self.test_loader
        indices = self.training_indices if training else self.test_indices
        labels = self.train_labels if training else self.test_labels
        np.random.shuffle(indices)

        complete = False
        batch_size, current_index = self.batch_size, 0
        while not complete:
            if current_index + batch_size >= dataset.shape[0]:
                batch_indices = np.concatenate((indices[current_index:], indices[:batch_size + current_index - dataset.shape[0]]))
                complete = True
            else:
                batch_indices = indices[current_index:current_index + batch_size]
                current_index += batch_size

            batch_indices = np.sort(batch_indices)

            spectrograms = torch.tensor(dataset[batch_indices], dtype=torch.float32)
            spectrograms = spectrograms.flatten(start_dim=1, end_dim=2) if self.model.flatten else spectrograms.unsqueeze(1)

            if lookup_labels and self._file_name_to_label is not None:
                batch_indices = self._file_name_to_label(labels[batch_indices])

            yield spectrograms, batch_indices  # No real labels passed, but ids to the spectrograms.

    def interactive_evaluation(self):
        self.model.eval()
        with torch.no_grad():
            print("Wait a moment while the first audio is being processed...")
            audio_model_interactive_evaluation(
                features=self.mode,
                test_loader=self.test_loader,
                test_labels=self.test_labels,
                # test_loader=self.train_loader,
                # test_labels=self.train_labels,
                model=self.model,
                loss_criterion=self._loss_criterion
            )
