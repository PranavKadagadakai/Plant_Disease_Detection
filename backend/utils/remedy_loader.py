import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REMEDY_DB_PATH = BASE_DIR / "data" / "remedies.json"


def load_remedies_json():
    """
    Safely load remedy database JSON.
    """

    if not REMEDY_DB_PATH.exists():
        raise FileNotFoundError(f"Remedy database not found: {REMEDY_DB_PATH}")

    with open(REMEDY_DB_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
