# Description: Questo script addestra un classificatore Random Forest per classificare audio deepfake e reali.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Legge il file CSV
data = pd.read_csv('extracted_features.csv')

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

# Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Salva il modello
#joblib.dump(model, 'deepfake_classifier.joblib')

# importanze delle caratteristiche
importances = model.feature_importances_

# indici delle caratteristiche più importanti
indices = np.argsort(importances)[::-1]

# nomi delle caratteristiche
feature_names = features[indices]

# Crea un grafico delle importanze delle caratteristiche
plt.figure(figsize=(10, 5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), feature_names, rotation=90)
plt.show()


shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        feature_names=feature_names,
    )

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
