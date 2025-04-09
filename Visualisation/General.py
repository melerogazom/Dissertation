import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def evaluate_classifier(csv_file, method_name):
    df = pd.read_csv(csv_file)
    
    # Drop non-numeric columns (like timestamps)
    df = df.select_dtypes(include=['number'])
    
    # Separate features (X) and labels (y)
    X = df.iloc[:, :-2]  # All columns except last two
    y = df.iloc[:, -1]   # Last column is the target (rep or not rep)
    
    # Convert labels to numeric values if necessary
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    train_sizes = np.linspace(0.1, 1.0, 10)  # Different training sizes
    f1_scores = []
    
    for size in train_sizes:
        if size == 1.0:
            # When train_size is 1.0, we don't specify train_size; just set test_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(size), random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    return train_sizes, f1_scores, method_name

def plot_f1_scores(signature_csv, handcrafted_csv):
    train_sizes_sig, f1_scores_sig, label_sig = evaluate_classifier(signature_csv, "Signature Methods")
    train_sizes_hand, f1_scores_hand, label_hand = evaluate_classifier(handcrafted_csv, "Handcrafted Methods")
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_sig, f1_scores_sig, marker='o', label=label_sig)
    plt.plot(train_sizes_hand, f1_scores_hand, marker='s', label=label_hand)
    plt.xlabel("Training Data Size")
    plt.ylabel("F1 Score")
    plt.title("Comparison of Signature and Handcrafted Methods")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
signature_csv = "signature_training_data.csv"
handcrafted_csv = "handcrafted_training_data.csv"
plot_f1_scores(signature_csv, handcrafted_csv)