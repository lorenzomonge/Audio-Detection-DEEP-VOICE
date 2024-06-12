import pandas as pd
import csv

def duplicate_last_n_rows(filepath, n=7, total_copies=50):
    # Legge il file CSV usando pandas
    df = pd.read_csv(filepath)
    
    # Ottiene le ultime n righe
    last_n_rows = df.tail(n)
    
    # Calcola quante volte duplicare l'intero blocco e quante righe aggiungere extra
    num_full_copies = total_copies // n
    extra_rows = total_copies % n
    
    # Crea copie delle ultime n righe
    duplicated_rows = pd.concat([last_n_rows] * num_full_copies, ignore_index=True)
    
    # Aggiunge le righe extra se necessario
    if extra_rows > 0:
        duplicated_rows = pd.concat([duplicated_rows, last_n_rows.head(extra_rows)], ignore_index=True)
    
    # Aggiunge le righe duplicate al dataframe originale
    df = pd.concat([df, duplicated_rows], ignore_index=True)
    
    # Salva il dataframe modificato nel file CSV
    df.to_csv(filepath, index=False)

# Esegue la funzione per duplicare le ultime 8 righe del file "extracted.csv" 50 volte
duplicate_last_n_rows("extracted_features_DEEP_SILENCE.csv", n=8, total_copies=50)

