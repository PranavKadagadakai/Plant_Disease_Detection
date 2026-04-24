import io

import torch
import torch.nn.functional as F
from PIL import Image

from config import DEVICE, MODEL_PATH
from models.model_builder import build_model
from utils.dataset import get_dataloaders
from utils.transforms import test_transform

model = None
classes = None


def load_model():
    global model, classes

    _, _, _, classes = get_dataloaders()

    model = build_model(len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()


def get_model():
    global model
    if model is None:
        load_model()
    return model


async def predict(file):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = test_transform(image).unsqueeze(0).to(DEVICE)

    model = get_model()

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_class = classes[int(pred_idx)]

        confidence, _ = torch.max(probs, dim=1)

        topk_probs, topk_indices = torch.topk(probs, k=3)

    return {
        "class_index": pred_idx,
        "class_name": pred_class,
        "confidence": float(confidence.item()),
        # "top_k": [
        #     {"class_name": classes[i], "probability": float(p)}
        #     for i, p in zip(topk_indices[0], topk_probs[0])
        # ],
    }
