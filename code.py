import pandas as pd
import numpy as np
import os
from obspy import read
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

# Detect if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Read Miniseed files from a specified directory
def read_mseed_files_from_directory(directory_path):
    mseed_files = [f for f in os.listdir(directory_path) if f.endswith('.mseed')]
    return mseed_files

# 2. Calculate slopes (velocity with respect to time)
def calculate_slopes(time_data, velocity_data):
    time_diff = np.diff(time_data)
    velocity_diff = np.diff(velocity_data)
    
    # Avoid division by zero when calculating slopes
    slopes = np.divide(velocity_diff, time_diff, where=(time_diff != 0))
    slopes = np.append(slopes, 0)  # Append 0 for the last value
    return slopes

# 3. Load Miniseed files and apply bandpass filters
def load_mseed_for_prediction(data_directory, filename):
    mseed_file = f'{data_directory}{filename}'
    st = read(mseed_file)
    tr = st[0]  # Take the first channel
    
    # Extract time and velocity data
    tr_times = tr.times()  # Relative time in seconds (time_sec)
    tr_data = tr.data      # Velocity data

    # Apply bandpass filter between 0.5 Hz and 1.0 Hz
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=0.5, freqmax=1.0)
    tr_filt = st_filt[0]
    tr_data_filt = tr_filt.data

    # Calculate slopes
    slopes = calculate_slopes(tr_times, tr_data_filt)

    # Return original times (time_sec), filtered velocity, and slopes
    return pd.DataFrame({
        'time_sec': tr_times,
        'velocity(m/s)': tr_data_filt,
        'slope': slopes
    })

# 4. Create a unified dataset using Miniseed files and the catalog
def create_mseed_dataset(catalog, data_directory):
    all_data = pd.DataFrame()

    for index, row in tqdm(catalog.iterrows(), total=catalog.shape[0], desc="Processing Miniseed files"):
        if index >= 5:  # Limit to first 5 entries for testing
            break
        filename = row['filename'] + '.mseed'
        time_abs = row['time_abs(%Y-%m-%dT%H:%M:%S.%f)']
        time_abs = datetime.strptime(time_abs, '%Y-%m-%dT%H:%M:%S.%f')
        
        # Load and process the Miniseed file
        seismic_data = load_mseed_for_prediction(data_directory, filename)
        
        # Label the earthquakes based on the 'time_abs' column
        seismic_data['label'] = np.where((seismic_data['time_sec'] >= time_abs.second - 5) &
                                         (seismic_data['time_sec'] <= time_abs.second + 5), 1, 0)
        
        # Add the processed data to the unified dataset
        all_data = pd.concat([all_data, seismic_data], ignore_index=True)
    
    return all_data

# 5. Create a neural network in PyTorch
class SismoNet(nn.Module):
    def __init__(self):
        super(SismoNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 3 features: time_sec, velocity, slope
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Binary output: earthquake or not
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x

# 6. Train the model with Mixed Precision Training in PyTorch
def train_model_mixed_precision(X_train, y_train, epochs=20, batch_size=32):
    # Convert training data to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model and move it to the device (GPU/CPU)
    model = SismoNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Initialize GradScaler for Mixed Precision
    scaler = GradScaler()

    # Train the model with Mixed Precision
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()  # Reset gradients

            # Forward pass with autocast for mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backpropagation with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    return model

# 7. Make predictions with the model on the GPU
def predict_pytorch(model, X_test):
    model.eval()  # Evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)  # Move data to GPU
        outputs = model(X_test_tensor)
        predictions = (outputs >= 0.5).float()  # Threshold of 0.5 for binary classification
    return predictions.cpu().numpy()  # Move back to CPU for saving or using

# 8. Process Miniseed test files for predictions and obtain only the filename and time_sec
def process_test_mseed_files(data_directory, mseed_files, model):
    predictions = []
    
    for filename in tqdm(mseed_files, desc="Making predictions on Miniseed files"):
        data = load_mseed_for_prediction(data_directory, filename)
        
        # Features for prediction
        X_test = data[['time_sec', 'velocity(m/s)', 'slope']]
        
        # Make predictions using the trained model
        predicted_labels = predict_pytorch(model, X_test)
        
        # Store only the filename and the time when an earthquake was detected
        for i, time in enumerate(data['time_sec']):
            if predicted_labels[i][0] == 1:  # Only store if an earthquake is detected
                predictions.append({
                    'filename': filename,
                    'time_sec': time
                })
    
    return predictions

# Main execution
if __name__ == "__main__":
    catalog_path = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
    data_directory = './data/lunar/training/data/S12_GradeA/'
    test_directory = './data/lunar/test/data/S15_GradeA/'
    
    # Read catalog and generate the dataset using Miniseed files
    catalog = pd.read_csv(catalog_path)
    dataset = create_mseed_dataset(catalog, data_directory)
    
    # Select features (slopes and velocity with respect to time) and labels (earthquakes)
    X = dataset[['time_sec', 'velocity(m/s)', 'slope']]  # Features
    y = dataset['label']  # Labels
    
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model in PyTorch with Mixed Precision
    model = train_model_mixed_precision(X_train, y_train, epochs=20, batch_size=32)
    
    # Read the Miniseed test files for making predictions
    mseed_files_test = read_mseed_files_from_directory(test_directory)
    test_predictions = process_test_mseed_files(test_directory, mseed_files_test, model)
    
    # Save predictions to a CSV file
    output_file = 'sismo_test_predictions_pytorch.csv'
    test_predictions_df = pd.DataFrame(test_predictions)
    test_predictions_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
