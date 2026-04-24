# Plant Disease Detection (Full Stack)

A full-stack AI application for detecting plant diseases using deep learning. The system includes a **PyTorch-based model**, a **FastAPI backend**, and a **React SPA frontend** for training, evaluation, and prediction with visualization support.

---

# 1. Overview

This project uses a **ResNet18 transfer learning model** trained on the PlantVillage dataset to classify plant diseases. It has been extended into a complete system with:

* Fully implemented FastAPI backend (no script reuse)
* React SPA frontend with routing
* Image upload + prediction
* Training and evaluation APIs
* Confusion matrix visualization (image rendering)

---

# 2. Features

## Machine Learning

* Transfer learning using ResNet18
* 15-class classification
* Train / Evaluate / Predict pipelines
* Metrics: accuracy, precision, recall, F1

## Backend (FastAPI)

* Re-implemented ML pipeline (no subprocess usage)
* `/train` → Train model
* `/evaluate` → Evaluate model + confusion matrix image
* `/predict` → Predict disease + confidence + top-k

## Frontend (React SPA)

* Multi-page routing (Train / Evaluate / Predict)
* Centered responsive UI
* Dark mode (default)
* Minimal design with orange primary theme
* Image preview + prediction results
* Confusion matrix full-width visualization

---

# 3. Dataset

The PlantVillage dataset contains 15 classes:

* Pepper: bacterial spot, healthy
* Potato: early blight, late blight, healthy
* Tomato: multiple diseases + healthy

---

# 4. Project Structure

```
project/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── routes/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/
│   │   └── styles/
│
├── PlantVillage/
├── plant_disease_model.pth
```

---

# 5. Requirements

## Backend

* Python >= 3.10
* PyTorch
* torchvision
* FastAPI
* matplotlib, seaborn, sklearn

## Frontend

* Node.js >= 18
* React + Vite
* axios
* react-router-dom

---

# 6. Setup Instructions

## Backend

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Runs at:

```
http://localhost:8000
```

---

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at:

```
http://localhost:5173
```

---

# 7. Usage

## Train

* UI → Train page → click Train
* API → POST `/train`

## Evaluate

* UI → Evaluate page
* Shows metrics + confusion matrix image
* API → GET `/evaluate`

## Predict

* UI → Upload image → Predict
* API → POST `/predict`

---

# 8. API Responses

## Train

* loss, accuracy, curves

## Evaluate

* accuracy, precision, recall, f1
* confusion_matrix_image (base64)

## Predict

* class_index
* class_name
* confidence
* top_k predictions

---

# 9. Model Architecture

* ResNet18 (ImageNet pretrained)
* Frozen backbone
* Custom classifier head
* CrossEntropyLoss + Adam

---

# 10. UI Highlights

* Dark mode default
* Orange primary actions
* Green for success states
* Clean card-based layout
* Full-width confusion matrix visualization

---

# 11. Engineering Decisions

* Reimplemented backend logic instead of reusing scripts
* Global model loading for performance
* Base64 image transfer for confusion matrix (stateless)
* React Router for multi-page navigation

---

# 12. Known Limitations

* Training is synchronous (blocking)
* No persistent dataset split
* Confusion matrix may be large for many classes

---

# 13. Future Improvements

* Async training (Celery / background tasks)
* Interactive confusion matrix (Plotly)
* Model versioning
* Deployment (Docker + cloud)

---

# 14. License

[Add your license information here]
