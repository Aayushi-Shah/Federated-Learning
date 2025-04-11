import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_generator import generate_labeled_temperature_data, create_windows_and_labels
from autoencoder import Autoencoder

# Set device (CPU for free-tier AWS instances)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Data Generation with 6-Month Simulation ---
# For 6 months (assuming 180 days), we need 180 * 1440 = 259200 samples.
data, labels = generate_labeled_temperature_data(num_samples=5000, noise_level=0.5, 
                                                   daily_anomaly_fraction=0.02, seasonal_anomaly_fraction=0.1,
                                                   downsample_rate=5)
window_size = 60  # 60-minute window size

# Create overlapping windows and window-level labels
windows, window_labels = create_windows_and_labels(data, labels, window_size)
print("Total windows:", windows.shape[0], "Anomalous windows:", np.sum(window_labels))

# Optionally, save windowed data to CSV for project presentation
import pandas as pd
df_windows = pd.DataFrame(windows)
df_windows["label"] = window_labels
df_windows.to_csv("synthetic_temperature_windows.csv", index=False)

# --- Train-Test Split (80-20 split) ---
dataset_size = windows.shape[0]
indices = np.arange(dataset_size)
np.random.shuffle(indices)
split = int(0.8 * dataset_size)
train_indices, test_indices = indices[:split], indices[split:]
X_train, y_train = windows[train_indices], window_labels[train_indices]
X_test, y_test = windows[test_indices], window_labels[test_indices]

# For unsupervised training, use only normal windows (label == 0) for training
normal_indices = np.where(y_train == 0)[0]
X_train_normal = X_train[normal_indices]

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_normal).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test = np.array(y_test)  # keeping y_test as a numpy array for metric calculations later

# Create DataLoader for training
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Model Initialization ---
model = Autoencoder(input_dim=window_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop with Early Stopping ---
num_epochs = 100
patience = 5  # Stop training if validation loss does not improve for 5 consecutive epochs
best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses = []

def evaluate_loss(model, data_tensor):
    model.eval()
    with torch.no_grad():
        inputs = data_tensor.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
    return loss.item()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_inputs, _ in train_loader:
        batch_inputs = batch_inputs.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_inputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_inputs.size(0)
    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)
    
    # Evaluate validation loss on test set
    val_loss = evaluate_loss(model, X_test_tensor)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict()  # Save best model state
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

# Load best model state after early stopping
model.load_state_dict(best_model_state)

# Plot training loss over epochs
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()

# --- Evaluation on Test Set ---
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    reconstructed = model(X_test_tensor)
    errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

# Set an anomaly threshold (tune based on error distribution)
threshold = 0.3
y_pred = (errors > threshold).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Ensure the models directory exists (it should be at the root level)
if not os.path.exists("models"):
    os.makedirs("models")

edge_name = os.getenv("HOSTNAME", "edge")
model_filename = f"models/autoencoder_model_{edge_name}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")

