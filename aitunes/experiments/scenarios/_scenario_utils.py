from os import path
from typing import Iterable, Union, final
from abc import abstractmethod, ABC

import os
import h5py
import numpy as np

from aitunes.audio_processing.preprocessing_collection import PreprocessingCollection
from aitunes.experiments.autoencoder_experiment import SpectrogramBasedAutoencoderExperiment
from aitunes.utils import download_and_extract
from aitunes.utils.audio_utils import AudioFeatures, precompute_spectrograms_for_audio_folder
from aitunes.experiments import AutoencoderExperiment


class ScenarioDescriptor:
    name: str  # Given through the decorator
    version: str  # Given through the decorator
    description: str  # Given through the decorator

    prod_grade: bool  # Given through the decorator
    prod_name: str  # Given through the decorator
    prod_desc: str  # Given through the decorator
    
    identifier: str  # Deduced from the name
    model_path: str  # Assigned when parsed by the TaskInitiator subclass
    history_path: str  # --
    _is_scenario: bool  # --

    def __call__(self, *args, **kwds): ...


def scenario(name: str, version: str = "1.0", description: Union[str, None] = None, prod_grade: bool = False, prod_name: str = "", prod_desc: str = "") -> ScenarioDescriptor:
    """
    Generates a decorator that will add additional information about the scenario

    Args:
        name (str): The name to display for the task on GUIs and in the console
        version (str, optional): To discriminate between similar experiments. Probably used alongside the display name. Defaults to "1.0".
        description (Union[str, None], optional): Additional information about the scenario (Why it was created, its features and expectations, ...). Defaults to None.
        prod_grade (bool, optional): Is the scenario production grade. Defaults to False.
        prod_name (str, optional): Name used to reference this model in production. Defaults to ""
        prod_desc (str, optional): Public description for this model. Defaults to ""

    Returns:
        ScenarioDescriptor: The function packed with all the information above
    """
    def decorator(f):
        f.name = name
        f.version = version
        f.description = description or "No description provided"
        f.identifier = name.lower().replace(" ", "_") + "_" + version
        f.prod_grade = prod_grade

        if prod_grade and not prod_name:
            raise Exception(f"Production grade scenario {name} {version} must provide a production name.")
        if prod_grade and not prod_desc:
            raise Exception(f"Production grade scenario {name} {version} must provide a production description.")

        f.prod_name = prod_name
        f.prod_desc = prod_desc
        f._is_scenario = True
        return f
    
    return decorator 


class ScenarioContainer(ABC):

    __all__ = ["_scenarios"]

    def __init__(self):
        super().__init__()
        os.makedirs(self.path_to_release_root, exist_ok=True)
        os.makedirs(self.path_to_history_root, exist_ok=True)

        # Recognize model functions
        self._scenarios: dict[str, ScenarioDescriptor] = { method.identifier: method for method in self.__class__.__dict__.values() if callable(method) and getattr(method, "_is_scenario", False) }
        for _s in self._scenarios.values():
            _s.model_path = path.join(self.path_to_release_root, _s.identifier + ".pth")  # Direct path to .pth file
            _s.history_path = path.join(self.path_to_history_root, _s.identifier)  # Direct path to root history folder
            os.makedirs(_s.history_path, exist_ok=True)
        

    @property
    @abstractmethod
    def path_to_history_root(self) -> str:
        """
        :return: Path to the history root folder where past models and training progress will be stored in dedicated folders
        """
        pass

    @property
    @abstractmethod
    def path_to_release_root(self) -> str:
        """
        :return: Path to release root where the final models will be saved for release
        """
        pass

    @property
    @abstractmethod
    def identifier(self) -> str:
        """
        :return: A unique identifier for the experiment 
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """
        :return: A description of the experiment (Purpose, ways it can be solved, approaches, ...)
        """
        pass

    @property
    @final
    def scenarios(self):
        return self._scenarios

    @abstractmethod
    def instantiate(self, s: ScenarioDescriptor, model_path: Union[str, bool, None]) -> AutoencoderExperiment:
        """
        Instantiate a task case with some scenario and returns it
        :param scenario: The scenario to instantiate
        :param model_path: An optional model path to override the default one used by the scenario (To load a previous model for example)
        """
        pass


class AudioBasedScenarioContainer(ScenarioContainer):

    @staticmethod
    def preprocess_audio(y: np.ndarray):
        """
        Preprocessing applied directly to the audio data before precomputing the spectrogram
        :param y: The raw audio data
        """
        return y
    
    @staticmethod
    def preprocess_spectrogram(y: np.ndarray):
        """
        Preprocessing applied to the spectrogram before being saved as a preprocessed numpy array
        :param y: The spectrogram data
        """
        return PreprocessingCollection.normalise(y, 0, 1)

    @property
    @abstractmethod
    def path_to_dataset_root(self) -> str:
        pass

    @property
    @abstractmethod
    def all_paths_to_training_spectrograms(self) -> tuple[str]:
        pass

    @property
    @abstractmethod
    def all_paths_to_eval_spectrograms(self) -> tuple[str]:
        pass

    @property
    @abstractmethod
    def path_to_audio_root(self) -> str:
        """
        Path to the folder root containing all audio files in the dataset.
        Will also be used to check if the dataset had indeed been downloaded
        """
        pass

    @property
    def path_to_training_spectrograms(self) -> Union[None, str]:
        if self.__current_mode is None:
            return None
        return self.all_paths_to_training_spectrograms[self.__current_mode]
    
    @property
    def path_to_eval_spectrograms(self) -> Union[None, str]:
        if self.__current_mode is None:
            return None
        return self.all_paths_to_eval_spectrograms[self.__current_mode]
    
    @property
    @abstractmethod
    def all_modes(self) -> tuple[AudioFeatures]:
        pass

    @property
    def batch_size(self) -> int:
        return 32
    
    @property
    def eval_proportions(self) -> float:
        return 0.05
    
    @property
    def mode(self) -> Union[None, AudioFeatures]:
        if self.__current_mode is None:
            return None
        return self.all_modes[self.__current_mode]
      
    @mode.setter
    def mode(self, mode_id: int):
        assert mode_id is None or isinstance(mode_id, int) and 0 <= mode_id < len(self.all_modes) 
        self.__current_mode = mode_id
        self._free_resources()
        if mode_id is not None:
            self._generate_datasets()
            self.__training_file = h5py.File(self.path_to_training_spectrograms, mode='r')
            self.__eval_file = h5py.File(self.path_to_eval_spectrograms, mode='r')
        
    @property
    @abstractmethod
    def dataset_info(self) -> tuple:
        """
        Get information about the dataset to download.
        :return: Target URL, download size, standalone directory
        """
        pass

    def __init__(self):
        super().__init__()
        self.__current_mode: int = 0  # Id to the currently selected mode
        self.__training_file: Union[h5py.File, None] = None
        self.__eval_file: Union[h5py.File, None] = None

    def instantiate(self, s, model_path):
        self.mode = None  # Datasets are generated when switching modes. See mode.setter
        model, loss, optimizer = s(self)
        assert self.mode is not None
        return SpectrogramBasedAutoencoderExperiment(self.identifier, model, model_path or s.model_path, loss, optimizer, self.__training_file, self.__eval_file, self.mode, self.batch_size, self.map_filename_to_label)

    def _free_resources(self):
        if self.__training_file is not None:
            self.__training_file.close()
            self.__training_file = None
        if self.__eval_file is not None:
            self.__eval_file.close()
            self.__eval_file = None
    
    def _create_audios(self):
        url, size, standalone_dir = self.dataset_info
        download_and_extract(
            url=url, target_path=self.path_to_dataset_root,
            zip_path=path.join(self.path_to_dataset_root, "..", self.identifier + ".zip"),
            final_size=size, standalone_zipped_dir=standalone_dir,
            clean=True
        )

    def _generate_datasets(self):  # Called from mode.setter
        if not path.exists(self.path_to_audio_root) and (not path.exists(self.path_to_training_spectrograms) or not path.exists(self.path_to_eval_spectrograms)):
            self._create_audios()
        precompute_spectrograms_for_audio_folder(self.path_to_audio_root, self.path_to_training_spectrograms, self.path_to_eval_spectrograms, self.eval_proportions, self.mode, self.preprocess_audio, self.preprocess_spectrogram)

    @abstractmethod
    def map_filename_to_label(self, filename: str|Iterable[str]) -> str | Iterable[str]:
        pass

    def __del__(self):
        self._free_resources()
