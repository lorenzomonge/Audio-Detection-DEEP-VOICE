# Funzione: Aggiunge le etichette bonafide e spoof al file CSV con le feature

import csv
import sys

def extract_labels_from_txt(txt_file):
    labels = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) > 3:
                label = 'bonafide' if 'bonafide' in parts[-1].strip() else 'spoof'
                labels.append(label)
    return labels

def add_labels_to_csv(csv_file, labels, output_csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    if len(rows) == 0:
        print("La lista delle righe è vuota.")
        return

    # Aggiunge l'intestazione "LABEL" come ultima colonna se non esiste
    header = rows[0]
    if "LABEL" not in header:
        header.append("LABEL")
    
    # Aggiunge le etichette alle righe del CSV
    label_index = 0
    for row in rows[1:]:
        if len(row) > 0:
            if label_index < len(labels):
                row.append(labels[label_index])
                label_index += 1
            else:
                row.append('sconosciuto')  # Gestisce il caso in cui ci sono più righe rispetto alle etichette
    
    # Scrive i dati modificati nel file CSV
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Percorsi dei file
txt_file = 'filtered_output_with_audio.txt'  # File di testo con le etichette
csv_file = sys.argv[1]  # File CSV con le feature
output_csv_file = 'features_with_labels_SILENCE.csv'  # File CSV di output

# Estrae le etichette dal file di testo
labels = extract_labels_from_txt(txt_file)

# Aggiunge le etichette al CSV
add_labels_to_csv(csv_file, labels, output_csv_file)

print(f"Etichette aggiunte e salvate in {output_csv_file}")
