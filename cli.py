import argparse
import os
import pickle
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Define Neural Network Models (same as in your training code)
class SLP(torch.nn.Module):
    def __init__(self, input_size):
        super(SLP, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

class SLMP(torch.nn.Module):
    def __init__(self, input_size):
        super(SLMP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

class MLMP(torch.nn.Module):
    def __init__(self, input_size):
        super(MLMP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 16)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# Function to preprocess data
def preprocess_data(df, scaler=None):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors='ignore')  # Drop irrelevant columns
    df['Geography'] = pd.factorize(df['Geography'])[0]  # Encoding categorical columns
    df['Gender'] = pd.factorize(df['Gender'])[0]  # Encoding categorical columns
    X = df.drop(columns=["Exited"])  # Features
    y = df["Exited"]  # Target column
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Standardize features
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, y, scaler

# Function to load saved model and scaler
def load_model_and_scaler(model_type='MLMP'):
    # Load the saved model
    with open(f"{model_type}_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load the saved scaler
    with open(f"{model_type}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Function to predict using the saved model
def predict(model, scaler, X):
    model.eval()  # Set model to evaluation mode
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
    predictions = (outputs > 0.5).float()  # Thresholding for binary classification
    return predictions.numpy()

# Main block to read the CSV and predict using the trained model
def main(file_path, model_type='MLMP'):
    # Load the model and scaler
    model, scaler = load_model_and_scaler(model_type=model_type)
    
    # Read and preprocess the data
    df = pd.read_csv(file_path)
    X, y, _ = preprocess_data(df, scaler=scaler)

    # Make predictions
    predictions = predict(model, scaler, X)

    # Evaluate the predictions
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    cm = confusion_matrix(y, predictions)

    # Display metrics
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# CLI Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Predict with a trained ML model")
    parser.add_argument('--data', type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument('--model', type=str, choices=['SLP', 'SLMP', 'MLMP'], default='MLMP', help="Choose the model type (SLP, SLMP, MLMP)")
    return parser.parse_args()

def cli():
    args = parse_args()

    # Verify that the file exists
    if not os.path.exists(args.data):
        print(f"Error: The file {args.data} does not exist.")
        return

    # Call the main function to start prediction
    main(args.data, model_type=args.model)

if __name__ == "__main__":
    cli()
