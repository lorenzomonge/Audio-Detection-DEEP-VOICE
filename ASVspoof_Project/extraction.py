# DESCRIZIONE: Questo script permette di filtrare le righe di un file di testo in base 
#a una lista di etichette specificate. 

file_path = "ASVspoof2019.LA.cm.eval.trl.txt"  # Sostituisci con il percorso effettivo del file

# Definisce le etichette da filtrare
labels_to_filter = ["A05", "A06", "A17", "A18", "A19", "bonafide"]

# Funzione per filtrare le righe in base alle etichette
def filter_lines(input_file, output_file, labels):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Filtra le righe che contengono una delle etichette specificate
    filtered_lines = [line for line in lines if any(label in line for label in labels)]
    
    # Scrive le righe filtrate nel file di output
    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)

input_file = 'ASVspoof2019.LA.cm.eval.trl.txt'
output_file = 'filtered_output.txt'


filter_lines(input_file, output_file, labels_to_filter)

print(f"Le righe filtrate sono state salvate in {output_file}")
