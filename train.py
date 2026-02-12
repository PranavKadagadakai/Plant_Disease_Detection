import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import build_model

# ---------------- CONFIG ----------------
DATA_DIR = "PlantVillage"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 224
MODEL_PATH = "plant_disease_model.pth"
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# Train / Validation split
# Create splits directory
os.makedirs("splits", exist_ok=True)

dataset_size = len(dataset)
indices = torch.randperm(dataset_size)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)

train_idx = indices[:train_size]
val_idx = indices[train_size : train_size + val_size]
test_idx = indices[train_size + val_size :]

# Save indices (IMPORTANT)
torch.save(train_idx, "splits/train_idx.pt")
torch.save(val_idx, "splits/val_idx.pt")
torch.save(test_idx, "splits/test_idx.pt")

# Create subsets
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
test_ds = Subset(dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = build_model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.fc.parameters(), lr=LR)

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Val Acc={acc:.2f}%")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved:", MODEL_PATH)
