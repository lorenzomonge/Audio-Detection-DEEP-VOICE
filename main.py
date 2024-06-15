import sys
import os
import csv
import time
from shared.feature_extraction import get_all_features_from_sample, SP_FEATS_NAMES
from pydub import AudioSegment
from tqdm import tqdm

def convert_audio_to_mono(filepath):
    """Converte il file audio in un canale mono."""
    audio = AudioSegment.from_wav(filepath)
    audio = audio.set_channels(1)
    audio.export(filepath, format=filepath.split(".")[-1])

def save_features_to_csv(filepath, features, header):
    """Salva le features estratte in un file CSV."""
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Scrive l'intestazione solo se il file non esiste già
        if not file_exists:
            writer.writerow(header)
        
        writer.writerow(features)

MODEL = "models/ADC_trained_model.sav"
FOLDER_PATH = input("Inserisci la cartella contenente i file audio: ")
OUTPUT_CSV = "extracted_features_ASVspoof_.csv"

# Uso SP_FEATS_NAMES come intestazione
header = SP_FEATS_NAMES

# Estrae le features per ogni file audio nella cartella
for root, dirs, files in os.walk(FOLDER_PATH):
    for file in tqdm(files, desc="Elaborazione dei file"):
        if file.endswith(".wav") or file.endswith(".flac") or file.endswith(".mp3"):  
            file_path = os.path.join(root, file)
            convert_audio_to_mono(file_path)  # Converto l'audio a mono
            
            start_time = time.time()  # Tempo di inizio estrazione feature
            
            try:
                sample_features = get_all_features_from_sample(file_path)
                # Verifica se sample_features non è vuoto e ha la lunghezza corretta
                save_features_to_csv(OUTPUT_CSV, sample_features, header)
            except Exception as e:
                print(f"Errore durante l'elaborazione del file {file_path}: {e}")
            
            end_time = time.time()  # Tempo di fine estrazione feature
            extraction_time = end_time - start_time  # Tempo totale di estrazione
            
            print(f"Tempo di estrazione per {file}: {extraction_time} secondi")
            
print(f"\nLe features del campione sono state salvate in {OUTPUT_CSV}")
