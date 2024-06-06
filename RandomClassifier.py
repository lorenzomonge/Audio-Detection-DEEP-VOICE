import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import shap# to explain the model
import joblib
import matplotlib.pyplot 


data = pd.read_csv('extracted_features.csv')
data = shuffle(data)


# replace "FAKE" with 0 and "REAL" with 1
data["LABEL"] = data["LABEL"].replace("FAKE", 0)
data["LABEL"] = data["LABEL"].replace("REAL", 1)



# get all columns except "LABEL" (gender)
features = data.keys()
features = features.drop("LABEL") # remove label

# Separating features and target
X = data.loc[:, features].values
y = data.loc[:, ['LABEL']].values.ravel()  

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42, shuffle=True)

# create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print("predict:", y_pred)
print("Accuracy:", accuracy)


# save the model
joblib.dump(model, 'deepfake_classifier.joblib')

#create an explainer with bar plot
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
matplotlib.pyplot.show()

