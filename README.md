# Plant Disease Detection

A PyTorch-based deep learning project for classifying plant diseases using transfer learning with ResNet18.

## Overview

This project uses a pre-trained ResNet18 model fine-tuned on the PlantVillage dataset to classify plant diseases across three crop types (pepper, potato, and tomato) with 15 different classes including healthy plants.

## Dataset

The PlantVillage dataset contains images organized into 15 classes:
- **Pepper**: Bacterial spot, healthy
- **Potato**: Early blight, healthy, late blight
- **Tomato**: 9 disease classes including bacterial spot, early/late blight, mosaic virus, yellow leaf curl virus, etc.

## Project Structure

```
├── model.py                 # ResNet18 model definition
├── train.py                 # Training script
├── evaluate.py              # Evaluation and metrics
├── predict.py               # Single image prediction
├── plant_disease_model.pth  # Saved model weights
├── sample_leaf.jpg          # Sample image for testing
├── PlantVillage/            # Dataset directory
├── splits/                  # Train/val/test indices
└── plots/                   # Generated visualizations
```

## Requirements

- Python >= 3.13
- PyTorch
- torchvision
- See `pyproject.toml` for full dependencies

## Setup

Using `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on the PlantVillage dataset:

```bash
python train.py
```

This will:
- Split data into 70% train, 15% validation, 15% test
- Train for 10 epochs (default)
- Save the model to `plant_disease_model.pth`
- Save split indices to `splits/` for reproducibility

### Evaluation

Evaluate the trained model on the test set:

```bash
python evaluate.py
```

Outputs accuracy, classification report, and confusion matrix visualization.

### Prediction

Predict the disease class for a single image:

```bash
python predict.py
```

Edit `IMAGE_PATH` in the script to change the input image.

## Model Architecture

- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Modification**: Fully connected layer replaced for 15-class classification
- **Training Strategy**: Transfer learning with frozen base layers
- **Optimizer**: Adam (training only the classifier)
- **Loss**: CrossEntropyLoss

## Results

The model achieves competitive accuracy on the PlantVillage test set. Run `evaluate.py` to see detailed metrics including:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix heatmap

## Development

See `AGENTS.md` for coding guidelines and project conventions.

## License

[Add your license information here]
