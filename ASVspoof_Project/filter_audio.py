# Funzione: Copia i file audio filtrati in una cartella di output



import os
import shutil
import sys
def filter_audio_files(txt_file, audio_dir, output_dir):
    # Legge i nomi dei file dal file di testo filtrato
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    # Estrae i nomi dei file audio dal file di testo
    audio_files = set()
    for line in lines:
        parts = line.split()
        if len(parts) > 1:
            audio_file = parts[1]
            audio_files.add(audio_file)
    
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Filtra e copia gli audio dalla cartella di input a quella di output
    for audio_file in os.listdir(audio_dir):
        file_name, file_extension = os.path.splitext(audio_file)
        if file_name in audio_files:
            src_path = os.path.join(audio_dir, audio_file)
            dst_path = os.path.join(output_dir, audio_file)
            shutil.copy(src_path, dst_path)
            print(f"Copied {audio_file} to {output_dir}")


txt_file = 'filtered_output.txt'  # Il file di testo filtrato
audio_dir = sys.argv[1]  # La cartella con i file audio originali passata come argomento
output_dir = 'filtered_audio'     # La cartella dove salvare i file audio filtrati


filter_audio_files(txt_file, audio_dir, output_dir)

print("Audio files filtering completed.")
