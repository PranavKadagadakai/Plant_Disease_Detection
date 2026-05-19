import re


def normalize_label(label: str) -> str:
    """
    Normalize model output labels to match
    remedies.json keys.

    Examples:
    Tomato_Late_blight
    -> tomato_late_blight

    Pepper__bell___Bacterial_spot
    -> pepper_bell_bacterial_spot
    """

    label = label.lower()

    label = re.sub(r"_+", "_", label)

    label = label.strip("_")

    label = label.replace(" ", "_")

    return label
