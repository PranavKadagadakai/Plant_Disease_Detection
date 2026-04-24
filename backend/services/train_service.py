import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, EPOCHS, LR, MODEL_PATH
from models.model_builder import build_model
from utils.dataset import get_dataloaders


def train_model():
    train_loader, val_loader, _, classes = get_dataloaders()

    model = build_model(len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history.append(epoch_loss)

    torch.save(model.state_dict(), MODEL_PATH)

    return {"status": "training completed", "epochs": EPOCHS, "final_loss": history[-1]}
