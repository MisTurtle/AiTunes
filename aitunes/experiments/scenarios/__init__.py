from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, scenario
from .linear_scenarios import LinearAugmentationScenarios
from .mnist_scenarios import MnistReconstructionScenarios
from .cifar10_scenarios import Cifar10ReconstructionScenarios
from .sinewave_scenarios import SinewaveReconstructionScenarios
from .gtzan_scenarios import GtzanReconstructionScenarios
from .fma_scenarios import FmaReconstructionScenarios

all_scenarios: dict[str, ScenarioContainer] = {
    cl.identifier: cl
    for cl in [
        LinearAugmentationScenarios(),
        MnistReconstructionScenarios(),
        Cifar10ReconstructionScenarios(),
        SinewaveReconstructionScenarios(),
        GtzanReconstructionScenarios(),
        FmaReconstructionScenarios()
    ]
}
