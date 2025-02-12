from .linear_scenarios import LinearAugmentationScenarios
from .mnist_scenarios import MnistReconstructionScenarios
from .sinewave_scenarios import SinewaveReconstructionScenarios
from .gtzan_scenarios import GtzanReconstructionScenarios
from .fma_scenarios import FmaReconstructionScenarios

all_scenarios = {
    cl.get_identifier(): cl() 
    for cl in [
        LinearAugmentationScenarios,
        MnistReconstructionScenarios,
        SinewaveReconstructionScenarios,
        GtzanReconstructionScenarios,
        FmaReconstructionScenarios
    ]
}
