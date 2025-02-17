import os
from typing import Any, Iterable, Union

# AiTunes imports
from aitunes.experiments import scenarios
from aitunes.experiments.autoencoder_experiment import AutoencoderExperiment
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, ScenarioDescriptor


class HeadlessActionPipeline:
    """
    Facilitator for experiment, scenario and model instantiation
    Could serve as an API endpoint for external programs using this package
    Mainly used to quickly scrap out some training program and by the GUI 
    """

    def __init__(self):
        self._selected_experiment: Union[ScenarioContainer, None] = None
        self._selected_scenario: Union[ScenarioDescriptor, None] = None
        self._selected_model: Union[str, None] = None

    def _require_set(self, attr: Any, msg: str):
        if attr is None:
            raise RuntimeError(msg)
    
    def _require_experiment(self):
        self._require_set(self._selected_experiment, "An experiment must be selected before performing this action")
    
    def _require_scenario(self):
        self._require_experiment()
        self._require_set(self._selected_scenario, "A scenario must be selected before performing this action")

    def describe_current_state(self) -> str:
        exp = self.get_selected_experiment()
        if exp is None:
            return "No experiment selected."
        
        description = f"Experiment: {exp.identifier}\n\t{exp.description}"
        scenario = self.get_selected_scenario()
        if scenario is None:
            return description
        
        description += f"\n\n\nScenario: {scenario.name} v{scenario.version}\n\t{scenario.description}"
        description += "\n\n\nModel: "
        model = self.get_selected_model()
        if model is not None and os.path.exists(model):
            description += f"{model}\tSize: {os.path.getsize(model) / 1e6:.2f}MB"
        else:
            models = self.list_models()
            if len(models) == 0:
                description += f"None found. Release model will be saved at {scenario.model_path} and training history at {scenario.history_path}"
            else:
                description += "None selected. Training will start from scratch."

        return description

    #
    # vvv LIST AND SELECT EXPERIMENTS vvv
    #
    def list_scripted_experiments(self) -> Iterable[str]:
        """
        :return: a list of all experiments which have a scenario interface
        """
        return list(scenarios.all_scenarios.keys())
    
    def select_experiment(self, identifier: str) -> bool:
        """
        :return: True if the change was successful, False otherwise
        """
        experiment = scenarios.all_scenarios.get(identifier, None)
        if experiment is None:
            return False
        self._selected_experiment = experiment
        self.close_scenario()
        return True
    
    def close_experiment(self):
        self._selected_experiment = None
        self.close_scenario()
        self.close_model()
    
    def get_selected_experiment(self) -> Union[ScenarioContainer, None]:
        return self._selected_experiment
    
    #
    # vvv LIST AND SELECT SCENARIOS vvv
    #
    def list_scenarios(self) -> Iterable[ScenarioDescriptor]:
        self._require_experiment()
        return list(self._selected_experiment.scenarios.values())

    def select_scenario(self, identifier: str) -> bool:
        self._require_experiment()
        
        if getattr(identifier, "identifier", None) is not None:
            # If a ScenarioDescriptor function is passed as a parameter, replace it with its identifier
            identifier = getattr(identifier, "identifier")

        s = self._selected_experiment._scenarios.get(identifier, None)
        if s is None:
            return False
        self._selected_scenario = s
        return True
    
    def close_scenario(self):
        self._selected_scenario = None
        self.close_model()
    
    def get_selected_scenario(self) -> Union[ScenarioDescriptor, None]:
        return self._selected_scenario

    #
    # vvv LIST AND SELECT RELEASE MODEL vvv
    #
    def has_release_model(self) -> bool:
        self._require_scenario()
        return os.path.exists(self._selected_scenario.model_path)
    
    def get_release_model(self) -> str:
        self._require_scenario()
        return self._selected_scenario.model_path
    
    def select_release_model(self) -> bool:
        self._require_scenario()
        if not self.has_release_model():
            return False
        self._selected_model = self._selected_scenario.model_path
        return True
    
    def close_model(self):
        self._selected_model = None
    
    #
    # vvv LIST AND SELECT AMONG OTHER MODELS
    #
    def _find_model_in(self, dir_path: str) -> Union[str, None]:
        """
        Checks if a particular folder contains at least one .pth model
        :return: The full path to the model if it exists, None otherwise
        """
        if not os.path.isdir(dir_path):
            return None
        for filename in os.listdir(dir_path):
            if filename.endswith(".pth"):
                return os.path.join(dir_path, filename)
        return None

    def list_models(self) -> Iterable[str]:
        self._require_scenario()
        
        paths = []
        if self.has_release_model():
            paths.append(self._selected_scenario.model_path)
        
        os.makedirs(self._selected_scenario.history_path, exist_ok=True)
        for filename in os.listdir(self._selected_scenario.history_path)[::-1]:  # More recent models come first
            model_path = self._find_model_in(os.path.join(self._selected_scenario.history_path, filename))
            if model_path is not None:
                paths.append(model_path)
        
        return paths

    def select_model(self, path: str) -> bool:
        self._require_scenario()
        if not path in self.list_models():
            return False
        self._selected_model = path
        return True
    
    def close_model(self):
        self._selected_model = None

    def get_selected_model(self) -> Union[str, None]:
        if self._selected_model is None or not os.path.exists(self._selected_model):
            return None
        return self._selected_model

    #
    # vvv Code generator to initiate the current state
    #
    def get_current_code(self, epochs: int, save_every: int, plotting: bool) -> str:
        if self._selected_experiment is None:
            return "# Code to initiate a scenario will be shown here once you select an experiment in the GUI"
        c = self.__class__
        imports = f"from {c.__module__} import {c.__name__}"
        lines = [
            f"actions = {c.__name__}()",
            f"actions.select_experiment('{self.get_selected_experiment().identifier}')"
        ]
        if self.get_selected_scenario() is not None:
            lines += [f"actions.select_scenario('{self.get_selected_scenario().identifier}')"]
        if self.get_selected_model() is not None:
            models = self.get_selected_model().replace('\\', '\\\\')
            lines += [f"actions.select_model('{models}')"]
        if self.get_selected_scenario() is not None and epochs > 0:
            lines += [f"actions.train({epochs}, {save_every}, {plotting})"]
        return imports + "\n" * 2 + "\n".join(lines)

    #
    # vvv Action launchers (train, eval, i_eval)
    #
    def instantiate_experiment(self) -> AutoencoderExperiment:
        self._require_scenario()
        return self.get_selected_experiment().instantiate(
            self.get_selected_scenario(),
            self.get_selected_model()
        )

    def train(self, epochs: int, save_period: int, plot_progress: bool):
        self._require_scenario()
        experiment = self.instantiate_experiment()
        experiment.save_every(save_period, self.get_selected_scenario().history_path)
        experiment.set_plotting(plot_progress)
        experiment.train(epochs)
    
    def evaluate(self):
        self._require_set(self._selected_model, "A model is required to be selected for it to be evaluated")
        experiment = self.instantiate_experiment()
        experiment.evaluate()
    
    def interactive_evaluation(self):
        self._require_set(self._selected_model, "A model is required to be selected for it to be evaluated interactively")
        experiment = self.instantiate_experiment()
        experiment.interactive_evaluation()