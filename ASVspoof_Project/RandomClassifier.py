import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import numpy as np
import joblib
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Leggi il file CSV
data = pd.read_csv(input("Enter the file path (only csv): "))

# Rimpiazza "spoof" con 0 e "bonafide" con 1
data["LABEL"] = data["LABEL"].replace("spoof", 0)
data["LABEL"] = data["LABEL"].replace("bonafide", 1)

# Rimuovi righe con valori NaN o infiniti
data = data.dropna()

# Ottieni tutte le colonne tranne la colonna "LABEL"
features = data.columns.drop("LABEL")
features = features.drop(["duration"])

# Separazione delle caratteristiche e del target
X = data[features].values
y = data["LABEL"].values

# Divisione del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42, shuffle=True)

# Creazione e addestramento del modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Salva il modello
#joblib.dump(model, 'deepfake_classifier2.joblib')

# Ottieni le importanze delle caratteristiche
importances = model.feature_importances_

# Ottieni gli indici delle caratteristiche più importanti
indices = np.argsort(importances)[::-1]

# Ottieni i nomi delle caratteristiche
feature_names = features[indices]

'''
# Crea un grafico delle importanze delle caratteristiche
plt.figure(figsize=(10, 5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), feature_names, rotation=90)
plt.show()
'''


print("model.classes_: ", model.classes_)
feature_names = [
    "duration", 
    "spectrum", 
    "mean_frequency", 
    "peak_frequency", 
    "frequencies_std", 
    "amplitudes_cum_sum",
    "mode_frequency", 
    "median_frequency", 
    "frequencies_q25", 
    "frequencies_q75",
    "iqr", 
    "freqs_skewness", 
    "freqs_kurtosis", 
    "spectral_entropy", 
    "spectral_flatness", 
    "spectral_centroid", 
    "spectral_bandwidth", 
    "spectral_spread", 
    "pectral_rolloff", 
    "energy",
    "rms", 
    "zcr", 
    "spectral_mean", 
    "spectral_rms", 
    "spectral_std", 
    "meanfun", 
    "minfun", 
    "maxfun", 
    "meandom", 
    "mindom", 
    "maxdom", 
    "dfrange", 
    "modindex"

]


shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", class_names = ["spoof", "bonafide"])


# matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualizza la matrice di confusione
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['spoof', 'bonafide'], rotation=45)
plt.yticks(tick_marks, ['spoof', 'bonafide'])

#etichette, titoli e una legenda
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
