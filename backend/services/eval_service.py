import base64
import io

import matplotlib

matplotlib.use("Agg")  # non-GUI backend

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import DEVICE, MODEL_PATH
from models.model_builder import build_model
from utils.dataset import get_dataloaders


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(16, 14))  # increased size

    sns.heatmap(
        cm, ax=ax, cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)  # higher DPI
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def evaluate_model():
    _, _, test_loader, classes = get_dataloaders()

    model = build_model(len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    preds, labels_list = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            labels_list.extend(labels.numpy())

    preds = np.array(preds)
    labels_list = np.array(labels_list)

    cm = confusion_matrix(labels_list, preds)

    cm_image = plot_confusion_matrix(cm, classes)

    return {
        "accuracy": float(accuracy_score(labels_list, preds)),
        "precision": float(precision_score(labels_list, preds, average="weighted")),
        "recall": float(recall_score(labels_list, preds, average="weighted")),
        "f1_score": float(f1_score(labels_list, preds, average="weighted")),
        "confusion_matrix_image": cm_image,
    }
