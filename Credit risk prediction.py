# ml_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/german_credit_data.csv")  # Update with your path
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

# ------------------- OUTLIER HANDLING ------------------- #
def cap_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    capped_col = np.where(col < lower, lower, np.where(col > upper, upper, col))
    return capped_col, lower, upper

for col in df.select_dtypes(include=np.number).columns:
    df[col] = cap_outliers(df[col])[0]

# ------------------- ENCODING ------------------- #
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Housing'] = df['Housing'].map({'own': 0, 'rent': 1, 'free': 2})
df['Saving accounts'] = df['Saving accounts'].map({'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3})
df['Saving accounts'] = df['Saving accounts'].fillna(0)
df['Checking account'] = df['Checking account'].map({'little': 0, 'moderate': 1, 'rich': 2})
df['Checking account'] = df['Checking account'].fillna(0)
df['Purpose'] = LabelEncoder().fit_transform(df['Purpose'])
df['Risk'] = df['Risk'].map({'good': 1, 'bad': 0})  # Target variable

# ------------------- FEATURE SELECTION ------------------- #
X = df.drop('Risk', axis=1)
y = df['Risk']

# Mutual Information (optional display)
mi_scores = mutual_info_classif(X, y, discrete_features='auto')
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Recursive Feature Elimination
scaler_rfe = StandardScaler()
X_scaled_rfe = scaler_rfe.fit_transform(X)

logreg = LogisticRegression(max_iter=5000, solver='liblinear')
rfe = RFE(logreg, n_features_to_select=8)
rfe.fit(X_scaled_rfe, y)

selected_features = X.columns[rfe.support_]
print("\nSelected Features by RFE:", selected_features.tolist())

# Final data
X_selected = X[selected_features]

# ------------------- TRAIN-TEST SPLIT & SCALING ------------------- #
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- MODEL TRAINING & EVALUATION ------------------- #
models = {
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_score = 0
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    score = model.score(X_test_scaled, y_test)
    results[name] = score
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, preds))

    if score > best_score:
        best_score = score
        best_model = model

print("\nModel Scores:", results)
print(f"\n✅ Best model: {type(best_model).__name__} with accuracy {best_score:.4f}")

# ------------------- SAVE MODEL, SCALER, SELECTED FEATURES ------------------- #
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/selected_features.pkl", "wb") as f:
    pickle.dump(selected_features.tolist(), f)

print("\n✅ Model, scaler, and selected features saved.")