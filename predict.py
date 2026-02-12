import torch
from PIL import Image
from torchvision import datasets, transforms

from model import build_model

IMAGE_PATH = "sample_leaf.jpg"
DATA_DIR = "PlantVillage"
MODEL_PATH = "plant_disease_model.pth"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load class names SAFELY from dataset
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes

img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

model = build_model(num_classes=15)  # update if needed
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

with torch.no_grad():
    output = model(img)
    pred_idx = torch.argmax(output, dim=1).item()
    pred_class = class_names[int(pred_idx)]

print(f"Predicted class index:{pred_idx}")
print(f"Predicted class name:{pred_class}")
