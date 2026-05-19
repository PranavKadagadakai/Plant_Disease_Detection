import io

import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    MODEL_PATH,
)
from models.model_builder import build_model
from utils.dataset import get_dataloaders
from utils.label_normalizer import (
    normalize_label,
)
from utils.remedy_loader import (
    load_remedies_json,
)
from utils.transforms import test_transform

model = None
classes = None

REMEDIES_DB = load_remedies_json()


def load_model():
    global model, classes

    _, _, _, classes = get_dataloaders()

    model = build_model(len(classes)).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )
    )

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

        pred_idx = torch.argmax(
            outputs,
            dim=1,
        ).item()

        raw_class = classes[int(pred_idx)]

        confidence, _ = torch.max(
            probs,
            dim=1,
        )

        topk_probs, topk_indices = torch.topk(
            probs,
            k=3,
        )

    response = format_prediction_response(
        raw_class,
        float(confidence.item()),
    )

    top_predictions = []

    for prob, idx in zip(
        topk_probs[0],
        topk_indices[0],
    ):
        raw_top_class = classes[int(idx)]

        normalized_top_class = normalize_label(raw_top_class)

        remedy_data = REMEDIES_DB.get(
            normalized_top_class,
            REMEDIES_DB["generic_fallback"],
        )

        top_predictions.append(
            {
                "raw_class": raw_top_class,
                "normalized_class": normalized_top_class,
                "confidence": float(prob.item()),
                "display_names": remedy_data["display_name"],
            }
        )

    return {
        "class_index": pred_idx,
        "raw_class_name": raw_class,
        "normalized_class_name": response["normalized_class_name"],
        "confidence": response["confidence"],
        "low_confidence_flag": response["low_confidence_flag"],
        "display_names": response["display_names"],
        "treatments": response["treatments"],
        "top_predictions": top_predictions,
    }


def format_prediction_response(
    class_name: str,
    confidence_score: float,
):
    """
    Format API response using normalized
    labels matching remedies.json.
    """

    normalized_class_name = normalize_label(class_name)

    low_confidence_flag = confidence_score < CONFIDENCE_THRESHOLD

    treatment_data = REMEDIES_DB.get(
        normalized_class_name,
        REMEDIES_DB["generic_fallback"],
    )

    return {
        "status": "success",
        "normalized_class_name": (normalized_class_name),
        "confidence": float(confidence_score),
        "low_confidence_flag": (low_confidence_flag),
        "display_names": treatment_data["display_name"],
        "treatments": treatment_data["remedies"],
    }
