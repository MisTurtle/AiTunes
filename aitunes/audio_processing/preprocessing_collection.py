import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

class PreprocessingCollection:

    @staticmethod
    def apply_padding(signal, num_expected_samples, mode="constant", padding_side="right"):
        """
        Applique un padding à un signal si sa longueur est inférieure à celle attendue.
        """

        if len(signal) < num_expected_samples:
  
            num_missing_samples = num_expected_samples - len(signal)

            if padding_side == "right":
                padded_signal = np.pad(signal, (0, num_missing_samples), mode=mode)
            elif padding_side == "left":
                padded_signal = np.pad(signal, (num_missing_samples, 0), mode=mode)
            else:
                raise ValueError
                
            return padded_signal
        return signal
    
    @staticmethod
    def normalise(array, min_val, max_val):
        """
        Normalise un tableau entre [min_val, max_val] en utilisant la normalisation Min-Max
        """
        if array.max() == array.min():
            return None
        norm_array = (array - array.min()) / (array.max() - array.min())   # [0, 1]
        norm_array = norm_array * (max_val - min_val) + min_val             # Exemple entre [min_val, max_val]
        return norm_array


    @staticmethod
    def denormalise(norm_array, original_min, original_max, norm_min=None, norm_max=None):
        """
        Denormalise un tableau à sa plage d'origine en utilisant la normalisation Min-Max inversée.
        """
        if norm_min is None:
            norm_min = norm_array.min()
        if norm_max is None:
            norm_max = norm_array.max()

        array = (norm_array - norm_min) / (norm_max - norm_min)
        array = array * (original_max - original_min) + original_min
        return array

    



