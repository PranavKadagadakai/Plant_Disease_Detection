import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model

# ---------------- CONFIG ----------------
DATA_DIR = "PlantVillage"  # change if needed
MODEL_PATH = "plant_disease_model.pth"
BATCH_SIZE = 32
IMG_SIZE = 224
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms (same as training)
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Load test indices
test_idx = torch.load("splits/test_idx.pt")

# Recreate test dataset
test_ds = Subset(dataset, test_idx)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


num_classes = len(dataset.classes)
class_names = dataset.classes

# Model
model = build_model(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

y_true = []
y_pred = []

# ---------------- EVALUATION ----------------
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
