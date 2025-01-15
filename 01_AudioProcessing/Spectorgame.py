import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def afficher_spectrogramme(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)  

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogramme de {audio_path}')
    plt.show()

# Fonction pour traiter tous les fichiers WAV dans un dossier
def afficher_spectrogrammes_dans_dossier(dossier):
    for fichier in os.listdir(dossier):
        if fichier.endswith('.wav'): 
            chemin_fichier = os.path.join(dossier, fichier)  
            afficher_spectrogramme(chemin_fichier)  

dossier_audio = "Samples/generated"  
afficher_spectrogrammes_dans_dossier(dossier_audio)