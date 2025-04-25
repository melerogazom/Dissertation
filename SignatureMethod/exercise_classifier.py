# === exercise classifier for the signature method ===

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load CSV
df = pd.read_csv("signature_training_data.csv")

# Ensure last two columns are: 'Exercise_Type' and 'Label'
# Keep only rows labeled as actual reps
df_rep = df[df['Label'] == 1]

# Extract labels
y = df_rep['Exercise_Type']

# Extract features
X = df_rep.drop(columns=['Timestamp', 'Exercise_Type', 'Label'])
X = X.select_dtypes(include=[np.number])

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier 
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate 
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save model 
joblib.dump(clf, "exercise_type_classifier.pkl")

# Reporting
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}\n")
print(f"Classification Report:\n{report}")