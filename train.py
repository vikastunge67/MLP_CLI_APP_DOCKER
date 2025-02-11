import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define Neural Network Models
class SLP(nn.Module):
    def __init__(self, input_size):
        super(SLP, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

class SLMP(nn.Module):
    def __init__(self, input_size):
        super(SLMP, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

class MLMP(nn.Module):
    def __init__(self, input_size):
        super(MLMP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# Function to preprocess data
def preprocess_data(df):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors='ignore')  # Drop irrelevant columns
    df['Geography'] = pd.factorize(df['Geography'])[0]  # Encoding categorical columns
    df['Gender'] = pd.factorize(df['Gender'])[0]  # Encoding categorical columns
    X = df.drop(columns=["Exited"])  # Features
    y = df["Exited"]  # Target column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize features
    return X_scaled, y, scaler

# Function to train models
def train_models(X, y, model_type='MLMP', epochs=100, batch_size=64, lr=0.001, scaler=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the data if not pre-scaled
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Compute class weights for imbalance handling
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoaders
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Select Model
    model_classes = {"SLP": SLP, "SLMP": SLMP, "MLMP": MLMP}
    model_class = model_classes.get(model_type, MLMP)
    model = model_class(input_size=X.shape[1])

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training Loop with Early Stopping
    best_loss = float('inf')
    patience = 5  # Early stopping patience
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Make predictions on the test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs > 0.5).float()

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    # Print metrics
    print(f"\nModel: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix: \n{cm}")

    # Save the model and scaler for future use
    with open(f"{model_type}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{model_type}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"{model_type} model and scaler have been saved.")

# Main block to read the CSV and train models
def main(file_path, model_type='MLMP'):
    df = pd.read_csv(file_path)
    X, y, scaler = preprocess_data(df)

    # Train and save the model
    train_models(X, y, model_type=model_type)

if __name__ == "__main__":
    main('Churn_Modelling.csv')  # Default dataset path
