# Description: Questo script addestra un classificatore Random Forest per classificare audio deepfake e reali.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
import shap
import numpy as np
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Legge il file CSV
file_path = input("Enter the file path (only csv): ")
data = pd.read_csv(file_path)

# Rimpiazza "FAKE" con 0 e "REAL" con 1
data["LABEL"] = data["LABEL"].replace("FAKE", 0)
data["LABEL"] = data["LABEL"].replace("REAL", 1)

#tutte le colonne tranne la colonna "LABEL"
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

shap.summary_plot(shap_values,
                  X_train,
                  feature_names=np.array(feature_names),
                  plot_type="bar",
                  class_names=["FAKE", "REAL"])

# Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


acc_score = accuracy
prec_score = precision_score(y_test, y_pred)
rec_score = recall_score(y_test, y_pred)
f1s = f1_score(y_test, y_pred)
MCCs = matthews_corrcoef(y_test, y_pred)
ROCareas = roc_auc_score(y_test, y_pred)

print("Mean results and (std.):\n")
print("Accuracy: " + str(round(np.mean(acc_score)*100, 3)) + "% (" + str(round(np.std(acc_score)*100, 3)) + ")")
print("Precision: " + str(round(np.mean(prec_score), 3)) + " (" + str(round(np.std(prec_score), 3)) + ")")
print("Recall: " + str(round(np.mean(rec_score), 3)) + " (" + str(round(np.std(rec_score), 3)) + ")")
print("F1-Score: " + str(round(np.mean(f1s), 3)) + " (" + str(round(np.std(f1s), 3)) + ")")
print("MCC: " + str(round(np.mean(MCCs), 3)) + " (" + str(round(np.std(MCCs), 3)) + ")")
print("ROC AUC: " + str(round(np.mean(ROCareas), 3)) + " (" + str(round(np.std(ROCareas), 3)) + ")")

# Salva il modello
joblib.dump(model, 'deepfake_classifier.joblib')

# importanze delle caratteristiche
importances = model.feature_importances_

# indici delle caratteristiche più importanti
indices = np.argsort(importances)[::-1]

# nomi delle caratteristiche
feature_names = features[indices]

'''
# Crea un grafico delle importanze delle caratteristiche
plt.figure(figsize=(10, 5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), feature_names, rotation=90)
plt.show()
'''

# matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualizza la matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
