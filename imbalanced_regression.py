import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Simulated Imbalanced Regression Dataset
class ImbalancedRegressionDataset(Dataset):
    def __init__(self, size=1000, imbalance_factor=10):
        self.x = np.random.rand(size, 1) * 10
        self.y = self.x ** 2 + np.random.randn(size, 1) * 3  # Quadratic with noise
        mask = self.y.flatten() < np.percentile(self.y, 100 - imbalance_factor)
        self.x, self.y = self.x[mask], self.y[mask]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

train_dataset = ImbalancedRegressionDataset()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Simple Regression Model
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train Initial Model
for epoch in range(10):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

# Generate Pseudo-Labels for Unlabeled Data
unlabeled_x = torch.rand(500, 1) * 10  # Simulated Unlabeled Data
with torch.no_grad():
    pseudo_labels = model(unlabeled_x).squeeze()

# Confidence-based Filtering
threshold = 2.0
uncertainty = torch.abs(pseudo_labels - pseudo_labels.mean())
confident_indices = uncertainty < threshold
confident_x, confident_y = unlabeled_x[confident_indices], pseudo_labels[confident_indices]

# Combine with Original Labeled Data
full_x = torch.cat([train_dataset[:][0], confident_x])
full_y = torch.cat([train_dataset[:][1], confident_y])
full_loader = DataLoader(list(zip(full_x, full_y)), batch_size=32, shuffle=True)

# Retrain Model
for epoch in range(10):
    for x_batch, y_batch in full_loader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

print("Training complete!")
