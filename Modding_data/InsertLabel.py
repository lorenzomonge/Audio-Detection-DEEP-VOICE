# Apre il file CSV
import csv

with open('extracted_features_DEEP_SILENCE.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Verifica se la lista rows è vuota o ha abbastanza elementi
if len(rows) == 0:
    print("La lista rows è vuota.")
elif len(rows) < 66:
    print("La lista rows non ha abbastanza elementi.")
    # Aggiunge righe vuote alla lista finché non ha abbastanza elementi
    while len(rows) < 66:
        rows.append([])

    # Aggiunge l'intestazione "LABEL" come ultima colonna
    rows[0].append("LABEL")

    # Imposta i valori di etichetta per l'audio falso e reale
    for i in range(0, 58):
        rows[i].append("FAKE")
    for i in range(58, 66):
        rows[i].append("REAL")

    # Scrive i dati modificati nel file CSV
    with open('extracted_features_DEEP_SILENCE.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
else:
    # La lista rows ha abbastanza elementi, nessuna azione necessaria
    pass
