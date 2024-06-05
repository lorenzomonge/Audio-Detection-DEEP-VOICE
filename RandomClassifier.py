import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Caricare il dataset CSV
data = pd.read_csv('extracted_features.csv')

# Sostituire le label 'fake' con 1 e 'real' con 0
data['LABEL'] = data['LABEL'].replace({'FAKE': 1, 'REAL': 0})

# Separare le features dai target
X = data.iloc[:,:-1] 
y = data.iloc[:,-1] 

# Divido il dataset in training e testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creo e alleno il modello Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Valuto il modello sul dataset di testing
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuratezza:", accuracy)

# Salvo il modello
joblib.dump(model, 'deepfake_classifier.joblib')
