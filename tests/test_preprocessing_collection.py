import unittest
import autoloader
import numpy as np
from aitunes.audio_processing import PreprocessingCollection
import torch
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt


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



    def apply_filters(self):
        file_path = 'C:\\Users\\hp\\Desktop\\Mon Project\\AiTunes\\assets\\Samples\\generated\\sine_asc_wave_150-300.wav'
        
        sample_rate = 5000  
        cutoff_freq = 200 
        filter_order = 7    
        
        audio_data, _ = librosa.load(file_path, sr=sample_rate)
        signal_tensor = torch.tensor(audio_data)
        
        filtered_signal_lowpass = PreprocessingCollection.apply_lowpass_filter(signal_tensor, sample_rate, cutoff_freq, filter_order)
        
        filtered_signal_highpass = PreprocessingCollection.apply_highpass_filter(signal_tensor, sample_rate, cutoff_freq, filter_order)
    

        def plot_fft(signal, ax, title):

            fft_signal = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(fft_signal), d=1/sample_rate)
            ax.plot(freqs[:len(freqs)//2], np.abs(fft_signal)[:len(fft_signal)//2])
            ax.set_title(title)
            ax.set_xlabel('Fréquence (Hz)')
            ax.set_ylabel('Amplitude')
            

        plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(3, 1, 1)
        plot_fft(signal_tensor.numpy(), ax1, 'signal original')
        ax2 = plt.subplot(3, 1, 2)
        plot_fft(filtered_signal_lowpass.numpy(), ax2, f'signal filtré passe-bas ({cutoff_freq} Hz)')
        ax3 = plt.subplot(3, 1, 3)
        plot_fft(filtered_signal_highpass.numpy(), ax3, f'signal filtré passe-haut ({cutoff_freq} Hz)')

        plt.tight_layout()
        plt.show()  
        input("")

if __name__ == '__main__':
    unittest.main()
