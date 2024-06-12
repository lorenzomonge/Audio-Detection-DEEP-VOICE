import sys
import os
import csv
import time
from shared.feature_extraction import get_all_features_from_sample, SP_FEATS_NAMES
from pydub import AudioSegment

def convert_audio_to_mono(filepath):
    """Converts the audio file to mono channel."""
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_channels(1)
    audio.export(filepath, format=filepath.split(".")[-1])

def save_features_to_csv(filepath, features, header):
    """Saves the extracted features to a CSV file."""
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header only if the file doesn't already exist
        if not file_exists:
            writer.writerow(header)
        
        writer.writerow(features)

MODEL = "models/ADC_trained_model.sav"
FOLDER_PATH = sys.argv[1]
OUTPUT_CSV = "extracted_features_ASVspoof_COMPLETE.csv"

# Uso SP_FEATS_NAMES come header
header = SP_FEATS_NAMES

# Estrae le features per ogni file audio nella cartella
for root, dirs, files in os.walk(FOLDER_PATH):
    for file in files:
        if file.endswith(".flac"):  
            file_path = os.path.join(root, file)
            convert_audio_to_mono(file_path)  # Converto l'audio a mono
            
            start_time = time.time()  # Tempo di inizio estrazione feature
            
            sample_features = get_all_features_from_sample(file_path)
            # Assumiamo che sample_features sia una lista
            save_features_to_csv(OUTPUT_CSV, sample_features, header)
            
            end_time = time.time()  # Tempo di fine estrazione feature
            extraction_time = end_time - start_time  # Tempo totale di estrazione
            
            print(f"Extraction time for {file}: {extraction_time} seconds")
            
          

print(f"\nSample features saved to {OUTPUT_CSV}")
