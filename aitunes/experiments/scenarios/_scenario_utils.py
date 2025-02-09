from os import path
import os
from typing import Union, final
from abc import abstractmethod, ABC

import torch

from aitunes.utils.audio_utils import AudioFeatures
from aitunes.experiments import AutoencoderExperiment


class ScenarioDescriptor:
    name: str  # Given through the decorator
    version: str  # Given through the decorator
    description: str  # Given through the decorator
    identifier: str  # Deduced from the name
    model_path: str  # Assigned when parsed by the TaskInitiator subclass
    history_path: str  # --
    _is_scenario: bool  # --

    def __call__(self, *args, **kwds): ...


def scenario(name: str, version: str = "1.0", description: Union[str, None] = None) -> ScenarioDescriptor:
    """
    Generates a decorator that will add additional information about the scenario
    :param name: The name to display for the task on GUIs and in the console
    :param version: To discriminate between similar experiments. Probably used alongside the display name
    :param description: Additional information about the scenario (Why it was created, its features and expectations, ...)
    """
    def decorator(f):
        f.name = name
        f.version = version
        f.description = description or "No description provided"
        f.identifier = name.lower().replace(" ", "_") + "_v" + version
        f._is_scenario = True
        return f
    
    return decorator 


class ScenarioContainer(ABC):

    __all__ = ["_scenarios"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Recognize model functions
        cls._scenarios: dict[str, ScenarioDescriptor] = {method.identifier: method for method in cls.__dict__.values() if callable(method) and getattr(method, "_is_scenario", False)}
        for _s in cls._scenarios.values():
            _s.model_path = path.join(cls.get_path_to_release_root(), _s.identifier + ".pth")  # Direct path to .pth file
            _s.history_path = path.join(cls.get_path_to_history_root(), _s.identifier)  # Direct path to root history folder
            os.makedirs(_s.history_path, exist_ok=True)

    def __init__(self):
        super().__init__()
        os.makedirs(self.get_path_to_history_root(), exist_ok=True)
        os.makedirs(self.get_path_to_release_root(), exist_ok=True)

    @staticmethod
    @abstractmethod
    def get_path_to_history_root() -> str:
        """
        :return: Path to the history root folder where past models and training progress will be stored in dedicated folders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_path_to_release_root() -> str:
        """
        :return: Path to release root where the final models will be saved for release
        """
        pass

    @staticmethod
    @abstractmethod
    def get_identifier() -> str:
        """
        :return: A unique identifier for the experiment 
        """

    @staticmethod
    @abstractmethod
    def get_description() -> str:
        """
        :return: A description of the experiment (Purpose, ways it can be solved, approaches, ...)
        """
        pass

    @final
    def get_scenarios(self):
        return self._scenarios

    @abstractmethod
    def instantiate(self, s: ScenarioDescriptor, model_path: Union[str, False, None]) -> AutoencoderExperiment:
        """
        Instantiate a task case with some scenario and returns it
        :param scenario: The scenario to instantiate
        :param model_path: An optional model path to override the default one used by the scenario (To load a previous model for example)
        """
        pass


class AudioBasedScenarioContainer(ScenarioContainer):

    def __init__(self):
        super().__init__()
        self.__current_mode = None  # High resolution or low resolution mode
    
    def get_mode(self) -> AudioFeatures:
        if self.__current_mode is None or not isinstance(self.__current_mode, AudioFeatures):
            raise ValueError("Audio-based scenarios must define some audio features settings before running.")
        return self.__current_mode
    
    def set_mode(self, m: AudioFeatures):
        self.__current_mode = m

