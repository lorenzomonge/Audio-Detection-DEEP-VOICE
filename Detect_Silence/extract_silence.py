import os
import sys
from pydub import AudioSegment
from pydub.silence import detect_silence
from tqdm import tqdm

def detect_and_save_silence(input_folder, output_folder, silence_thresh=-50, min_silence_len=10):
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)
    
    # Processa ogni file audio nella cartella di input
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.wav') or filename.endswith('.flac') or filename.endswith('.mp3'):
            audio_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_file(audio_path)
            
            # Trova segmenti di silenzio (ritorna intervalli di tempo)
            silence_ranges = detect_silence(
                audio, 
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            # Se non ci sono segmenti di silenzio, salta il file
            if not silence_ranges:
                print(f"Nessun segmento di silenzio trovato in {filename}")
                continue
            
            # Unisce tutti i segmenti di silenzio in un unico audio
            silence_audio = AudioSegment.empty()
            for start, end in silence_ranges:
                silence_audio += audio[start:end]
            
            # Salva l'audio unito nella cartella di output
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
            silence_audio.export(output_path, format="wav")
           

input_folder = input("Enter input folder: ") #inserire il percorso della cartella contenente i file audio
output_folder = input("Enter output folder: ")#inserire il percorso della cartella di output

detect_and_save_silence(input_folder, output_folder)
