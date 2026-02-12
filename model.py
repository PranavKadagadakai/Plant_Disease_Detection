import torch.nn as nn
from torchvision import models


def build_model(num_classes):
    model = models.resnet18(weights="DEFAULT")

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
