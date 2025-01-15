import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal
import noisereduce as nr  # Pour la réduction de bruit

def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        

#Prétaitement 

# Filtrage des fréquences indésirables
def filtre_audio(y, sr, lowcut=300.0, highcut=8000.0):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(1, [low, high], btype='band')
    y_filtree = scipy.signal.filtfilt(b, a, y)
    return y_filtree

# Normalisation de l'amplitude
def normaliser_amplitude(y):
    # Normalisation de l'amplitude entre -1 et 1
    y_normalized = librosa.util.normalize(y)
    return y_normalized

# Réduction du bruit (en utilisant le package noisereduce)
def reduction_bruit(y, sr):
    y_reduced = nr.reduce_noise(y=y, sr=sr)
    return y_reduced

# Segmenter l'audio en fenêtres
def segmenter_audio(y, sr, taille_fenetre=2048, hop_size=512):
    return librosa.util.frame(y, frame_length=taille_fenetre, hop_length=hop_size)


# Calcul du spectrogramme logarithmique
def calculer_spectrogramme_log(y, sr):
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    return D

# Fonction pour le prétraitement d'un fichier audio et génération du spectrogramme
def preprocess_audio(audio_path, output_folder, lowcut=300.0, highcut=8000.0):
    y, sr = librosa.load(audio_path, sr=None)  # Chargement du fichier audio
    
    # Appliquer les différents prétraitements
    y = normaliser_amplitude(y)  # Normaliser l'amplitude
    y = filtre_audio(y, sr, lowcut, highcut)  # Filtrage des fréquences
    y = reduction_bruit(y, sr)  # Réduction du bruit

    spectrogramme = calculer_spectrogramme_log(y, sr)  # Calcul du spectrogramme

    # Sauvegarder le spectrogramme sous forme d'image
    nom_fichier = os.path.join(output_folder, f"{os.path.basename(audio_path)}_spectrogram.png")
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogramme, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogramme de {audio_path}')
    plt.savefig(nom_fichier)
    plt.close()

    print(f"Spectrogramme sauvegardé à : {nom_fichier}")

def preprocess_all_files_in_folder(dossier, output_folder):
    create_output_folder(output_folder)  # Créer le dossier de sortie si nécessaire
    
    for fichier in os.listdir(dossier):
        if fichier.endswith('.wav'):  # Si le fichier est un WAV
            chemin_fichier = os.path.join(dossier, fichier)
            preprocess_audio(chemin_fichier, output_folder)  # Traitement du fichier audio

def main():
    dossier_audio = "Samples/generated"  # Dossier contenant les fichiers audio
    output_folder = "Samples/processed_spectrograms"  # Dossier pour sauvegarder les spectrogrammes

    print("Démarrage du prétraitement des fichiers audio...")
    preprocess_all_files_in_folder(dossier_audio, output_folder)
    print("Prétraitement terminé.")

if __name__ == "__main__":
    main()
