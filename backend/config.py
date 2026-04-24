import torch

DATA_DIR = "PlantVillage"
MODEL_PATH = "models/plant_disease_model.pth"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
