import os

# Questo script filtra le righe di un file di testo in base a determinate etichette e file audio corrispondenti.

# Definisce le etichette da filtrare
labels_to_filter = ["A05", "A06", "A17", "A18", "A19", "bonafide"]

# Funzione per filtrare le righe in base alle etichette
def filter_lines(input_file, output_file, labels, audio_dir):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Ottiene la lista dei file audio nella directory
    audio_files = {os.path.splitext(audio_file)[0] for audio_file in os.listdir(audio_dir)}
    
    # Filtra le righe che contengono una delle etichette specificate e corrispondono ai file audio
    filtered_lines = [
        line for line in lines
        if any(label in line for label in labels) and line.split()[1] in audio_files
    ]
    
    # Scrive le righe filtrate nel file di output
    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)


input_file = 'ASVspoof2019.LA.cm.eval.trl.txt'  # File di testo originale
output_file = 'filtered_output_with_audio.txt'  # Nuovo file di testo con le righe filtrate
audio_dir = 'filtered_audio'  # Cartella con i file audio filtrati


filter_lines(input_file, output_file, labels_to_filter, audio_dir)

print(f"Le righe filtrate con audio sono state salvate in {output_file}")
