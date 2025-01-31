import unittest
import autoloader
import numpy as np
from aitunes.audio_processing import PreprocessingCollection

class TestPreprocessingCollection(unittest.TestCase):

    def test_apply_padding_right(self):
        signal = np.array([1, 2, 3])
        num_expected_samples = 6
        padded_signal = PreprocessingCollection.apply_padding(signal, num_expected_samples, mode="constant", padding_side="right")
        
        print("Signal original :", signal)
        print("Signal après padding à droite:", padded_signal)
        
        self.assertEqual(len(padded_signal), num_expected_samples)
        self.assertTrue(np.array_equal(padded_signal[-2:], [0, 0])) 

    def test_apply_padding_left(self):

        signal = np.array([1, 2, 3])
        num_expected_samples = 6
        padded_signal = PreprocessingCollection.apply_padding(signal, num_expected_samples, mode="constant", padding_side="left")
        
        print("Signal original :", signal)
        print("Signal après padding à gauche:", padded_signal)
        
        self.assertEqual(len(padded_signal), num_expected_samples)
        self.assertTrue(np.array_equal(padded_signal[:2], [0, 0])) 


    def test_normalise(self):

        array = np.array([1, 2, 3, 4, 5])
        min_val = 0
        max_val = 10
        norm_array = PreprocessingCollection.normalise(array, min_val, max_val)
        
        print("Tableau original", array)
        print("Tableau normalisé:", norm_array)
        
        self.assertEqual(norm_array.min(), min_val)
        self.assertEqual(norm_array.max(), max_val)

    def test_denormalise(self):

        norm_array = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        norm_min = 0.0
        norm_max = 1.0
        original_min = 0
        original_max = 10
        array = PreprocessingCollection.denormalise(norm_array, original_min, original_max, norm_min=norm_min, norm_max=norm_max)
        
        print("Tableau normalisé:", norm_array)
        print("Tableau dénormalisé:", array)
        
        self.assertEqual(array.min(), original_min)
        self.assertEqual(array.max(), original_max)

   

if __name__ == '__main__':
    unittest.main()
