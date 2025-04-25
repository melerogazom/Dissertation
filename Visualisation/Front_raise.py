# === Front Raise ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def evaluate_classifier(csv_file, method_name, exercise):
    # Load and filter data for the specific exercise
    df = pd.read_csv(csv_file)
    
    # Make sure the Exercise_Type column exists
    if 'Exercise_Type' not in df.columns:
        raise ValueError(f"CSV file must contain an 'Exercise_Type' column, but '{csv_file}' does not.")

    # Filter data for the current exercise
    exercise_df = df[df['Exercise_Type'] == exercise]
    
    # Drop non-numeric columns (like timestamps and exercise type)
    exercise_df = exercise_df.select_dtypes(include=['number'])
    
    # Separate features (X) and labels (y)
    X = exercise_df.iloc[:, :-2]  # All columns except last two
    y = exercise_df.iloc[:, -1]   # Last column is the target (rep or not rep)
    
    # Convert labels to numeric values if necessary
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Evaluate model using different training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    f1_scores = []

    for size in train_sizes:
        if size == 1.0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(size), random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    return train_sizes, f1_scores, method_name

def plot_f1_scores(signature_csv, handcrafted_csv):
    df_sig = pd.read_csv(signature_csv)  # Load the signature CSV to get the exercise types
    exercise_types = df_sig['Exercise_Type'].unique()  # Get unique exercise types
    
    # Make sure we have both signature and handcrafted data containing the same exercise types
    df_hand = pd.read_csv(handcrafted_csv)
    hand_exercise_types = df_hand['Exercise_Type'].unique()
    
    # Check if both files contain the same exercise types
    common_exercises = np.intersect1d(exercise_types, hand_exercise_types)
    if len(common_exercises) == 0:
        raise ValueError("No common exercise types found between signature and handcrafted datasets.")
    
    # Iterate through each exercise type and plot F1 scores for both methods
    print(f"Processing {common_exercises[1]}...")  # For debugging
        
    # Evaluate signature and handcrafted methods for each exercise
    train_sizes_sig, f1_scores_sig, label_sig = evaluate_classifier(signature_csv, "Signature Methods", common_exercises[1])
    train_sizes_hand, f1_scores_hand, label_hand = evaluate_classifier(handcrafted_csv, "Handcrafted Methods", common_exercises[1])
        
    # Plot the F1 scores for each exercise type
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_sig, f1_scores_sig, marker='o', label=label_sig)
    plt.plot(train_sizes_hand, f1_scores_hand, marker='s', label=label_hand)
    plt.xlabel("Training Data Size")
    plt.ylabel("F1 Score")
    plt.title(f"Comparison of Signature and Handcrafted Methods for {common_exercises[1]}")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
signature_csv = "signature_training_data.csv"
handcrafted_csv = "handcrafted_training_data.csv"
plot_f1_scores(signature_csv, handcrafted_csv)