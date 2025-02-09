from .linear_scenarios import LinearAugmentationScenarios
from .mnist_scenarios import MnistReconstructionScenarios
from .sinewave_scenarios import SinewaveReconstructionScenarios


all_scenarios = {
    cl.get_identifier(): cl() 
    for cl in [
        LinearAugmentationScenarios,
        MnistReconstructionScenarios,
        SinewaveReconstructionScenarios
    ]
}
