import numpy as np
import matplotlib.pyplot as plt

duration = 1.0
high_res = 1000
sample_rate = 100

t_cont = np.linspace(0, duration, high_res)
t_samp = np.linspace(0, duration, sample_rate, endpoint=False)

freqs = [3, 7, 13]
analog_signal = sum(np.sin(2 * np.pi * f * t_cont) for f in freqs)
sampled_signal = sum(np.sin(2 * np.pi * f * t_samp) for f in freqs)

plt.figure(figsize=(12, 4))
plt.plot(t_cont, analog_signal, label="Signal analogique", color='blue')
markerline, stemline, baseline = plt.stem(t_samp, sampled_signal, linefmt='r-', markerfmt='ro', basefmt=' ', label="Échantillons")
plt.setp(markerline, markersize=4)
plt.title("Illustration du processus d'échantillonnage")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
