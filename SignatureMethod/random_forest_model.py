# === Classifier for the signature method ===

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# load Data 
df = pd.read_csv("signature_training_data.csv")

# shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# display class distribution 
print(f"Class Distribution: {df['Label'].value_counts().to_dict()}")
print(f"Total samples in dataset: {df.shape[0]}")

# Ensure last two columns are: 'Exercise_Type' 
if 'Exercise_Type' not in df.columns:
    raise ValueError("CSV file must contain an 'Exercise_Type' column.")

# split sata by exercise type
exercise_types = df['Exercise_Type'].unique()

# Minimum number of samples required per exercise type
min_samples = 5  

for exercise in exercise_types:
    exercise_df = df[df['Exercise_Type'] == exercise]
    
    # Skip exercises with too few samples
    if exercise_df.shape[0] < min_samples:
        print(f"Skipping {exercise} due to insufficient samples ({exercise_df.shape[0]} samples).")
        continue

    # Extract features and labels
    X = exercise_df.iloc[:, 1:-2].values  # All columns except timestamp, exercise type and label
    y = exercise_df['Label'].values      # Last column (rep/non-rep)

    # Train/test split 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Train a classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Shows precision, recall, f1-score, and support for each class (0 and 1)
    print(f"\nModel Accuracy for {exercise}: {accuracy:.4f}")
    print(f"\nClassification Report for {exercise}:")
    print(classification_report(y_test, y_pred))

    # Save model for each exercise
    model_filename = f'activity_classifier_{exercise}.pkl'
    joblib.dump(clf, model_filename)
    print(f"Model saved to {model_filename}")