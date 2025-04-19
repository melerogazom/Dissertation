import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ==== Load Data ====
CSV_FILE = "signature_training_data.csv"
df = pd.read_csv(CSV_FILE)

# ==== Shuffle Data ====
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==== Display Class Distribution ====
print(f"🔢 Class Distribution: {df['Label'].value_counts().to_dict()}")
print(f"Total samples in dataset: {df.shape[0]}")

# ==== Ensure 'Exercise_Type' Exists ====
if 'Exercise_Type' not in df.columns:
    raise ValueError("CSV file must contain an 'Exercise_Type' column.")

# ==== Split Data by Exercise Type ====
exercise_types = df['Exercise_Type'].unique()

# Minimum number of samples required per exercise type
min_samples = 5  

for exercise in exercise_types:
    exercise_df = df[df['Exercise_Type'] == exercise]
    
    # Skip exercises with too few samples
    if exercise_df.shape[0] < min_samples:
        print(f"⚠️ Skipping {exercise} due to insufficient samples ({exercise_df.shape[0]} samples).")
        continue

    # Extract Features & Labels
    X = exercise_df.iloc[:, 1:-2].values  # All columns except timestamp, exercise type & label
    y = exercise_df['Label'].values      # Last column (rep/non-rep)

    # ==== Stratified Split (Ensures class balance in both train/test) ====
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # ==== Train a Classifier ====
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ==== Evaluate Model ====
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model Accuracy for {exercise}: {accuracy:.4f}")
    print(f"\n📊 Classification Report for {exercise}:")
    print(classification_report(y_test, y_pred))

    # ==== Save Model for Each Exercise ====
    model_filename = f'activity_classifier_{exercise}.pkl'
    joblib.dump(clf, model_filename)
    print(f"📝 Model saved to {model_filename}")