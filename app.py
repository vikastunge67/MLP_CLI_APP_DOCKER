import os
from flask import Flask, render_template, request
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__, template_folder='my_templates')  # Template folder is my_templates

# Define a Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Preprocess the uploaded data
def preprocess_data(df):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors='ignore')  # Drop unnecessary columns
    df['Geography'] = pd.factorize(df['Geography'])[0]  # Encoding categorical columns
    df['Gender'] = pd.factorize(df['Gender'])[0]  # Encoding categorical columns
    X = df.drop(columns=["Exited"])  # Features
    y = df["Exited"]  # Target column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize features
    return X_scaled, y, scaler

# Train the Neural Network Model
def train_model(X, y, scaler):
    model = SimpleNN(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # Save the trained model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and model training
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    # Read and preprocess the CSV data
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f'Error reading the CSV file: {e}'

    X, y, scaler = preprocess_data(df)
    
    # Train the model and save it
    try:
        train_model(X, y, scaler)
    except Exception as e:
        return f'Error during model training: {e}'

    return 'Model trained successfully!'

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
