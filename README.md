# Plant Disease Detection (Full Stack)

A full-stack AI-powered plant disease detection system built with **PyTorch**, **FastAPI**, and **React**. The application provides disease prediction, model training, evaluation metrics, multilingual treatment recommendations, and a farmer-focused advisory workflow with low-confidence escalation support.

---

# 1. Overview

This project uses a **ResNet18 transfer learning model** trained on the PlantVillage dataset to classify plant diseases across multiple crop categories.

The system has evolved into a production-oriented full-stack platform featuring:

* FastAPI backend with modular ML services
* React SPA frontend with responsive dashboard UI
* Disease prediction with confidence scoring
* Localized remedy recommendations
* Multilingual interface support
* Light/Dark accessibility themes
* Training and evaluation pipelines
* Confusion matrix visualization
* Low-confidence expert escalation workflow

---

# 2. Features

## Machine Learning

* Transfer learning using ResNet18
* PlantVillage dataset integration
* Multi-class disease classification
* Train / Evaluate / Detect workflows
* Accuracy, Precision, Recall, and F1 metrics
* GPU-aware model execution support

## Backend (FastAPI)

* Fully modular service architecture
* Native ML inference pipeline (no subprocess execution)
* Structured API responses
* Remedy lookup system using JSON-based localized datasets
* Confidence threshold engine for uncertain predictions
* REST endpoints for training, evaluation, and prediction

### API Endpoints

* `POST /train` → Train model
* `GET /evaluate` → Evaluate trained model
* `POST /detect` → Detect plant disease from uploaded image
* `GET /` → API health check

## Frontend (React + Vite)

* Responsive dashboard interface
* React Router SPA navigation
* Image upload with preview support
* Confidence visualization components
* Detection result cards
* Dynamic treatment recommendation UI
* Kannada, Hindi, and English language support
* Light mode and dark mode theme system
* Mobile-friendly layout and typography
* Dedicated landing/home page with responsive hero section
* CTA navigation to disease detection workflow
* Feature overview cards on homepage
* Clickable home branding in navbar

## Farmer Advisory Enhancements

* Multilingual treatment recommendations
* Organic, Chemical, and Cultural remedy categorization
* Low-confidence diagnostic warning system
* Expert escalation workflow for uncertain predictions
* Outdoor-readable UI theme support

---

# 3. Dataset

The model is trained using the **PlantVillage dataset**.

Supported disease categories include:

* Pepper Bell

  * Bacterial Spot
  * Healthy

* Potato

  * Early Blight
  * Late Blight
  * Healthy

* Tomato

  * Bacterial Spot
  * Early Blight
  * Late Blight
  * Leaf Mold
  * Septoria Leaf Spot
  * Spider Mites
  * Target Spot
  * Mosaic Virus
  * Yellow Leaf Curl Virus
  * Healthy

---

# 4. Tech Stack

## Backend

* Python 3.13
* FastAPI
* PyTorch
* TorchVision
* TensorFlow
* Scikit-learn
* Matplotlib
* Seaborn
* Pillow
* Uvicorn

## Frontend

* React
* Vite
* Axios
* React Router DOM
* Context API
* CSS

---

# 5. Project Structure

```text
pranavkadagadakai-plant_disease_detection/
├── README.md
├── AGENTS.md
│
├── backend/
│   ├── app.py
│   ├── main.py
│   ├── config.py
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── data/
│   │   └── remedies.json
│   ├── models/
│   │   └── model_builder.py
│   ├── routes/
│   │   ├── detect.py
│   │   ├── evaluate.py
│   │   └── train.py
│   ├── services/
│   │   ├── detection_service.py
│   │   ├── eval_service.py
│   │   └── train_service.py
│   └── utils/
│       ├── dataset.py
│       ├── label_normalizer.py
│       ├── remedy_loader.py
│       └── transforms.py
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── api/
        │   └── api.js
        ├── components/
        │   ├── ConfidenceBar.jsx
        │   ├── Detect.jsx
        │   ├── DetectionResult.jsx
        │   ├── Evaluate.jsx
        │   ├── Home.jsx
        │   ├── LanguageCard.jsx
        │   ├── Navbar.jsx
        │   ├── Train.jsx
        │   └── TreatmentCard.jsx
        ├── context/
        │   ├── LanguageContext.jsx
        │   └── ThemeContext.jsx
        ├── i18n/
        │   └── i18n.js
        ├── locales/
        │   ├── en.json
        │   ├── hi.json
        │   └── kn.json
        ├── pages/
        │   └── Dashboard.jsx
        └── styles/
            └── app.css
```

---

# 6. Installation

## Clone Repository

```bash
git clone <repository-url>
cd Plant_Disease_Detection
```

---

# 7. Backend Setup

## Create Environment

```bash
cd backend
```

Using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Start Backend Server

```bash
uvicorn app:app --reload
```

Backend runs at:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

---

# 8. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

---

# 9. Usage

## Home / Landing Page

Frontend:

* Responsive landing page with project overview
* Quick navigation to disease detection workflow
* Feature highlights and multilingual support overview
* Theme-aware UI compatible with light and dark modes

## Train Model

Frontend:

* Open Train page
* Start model training
* Monitor metrics and training progress

API:

```http
POST /train
```

---

## Evaluate Model

Frontend:

* Open Evaluate page
* View metrics and confusion matrix visualization

API:

```http
GET /evaluate
```

Returns:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix Image

---

## Detect Disease

Frontend:

* Upload a plant leaf image
* Run disease detection
* View confidence score and treatment recommendations

API:

```http
POST /detect
```

Returns:

* Predicted disease
* Confidence score
* Localized display names
* Treatment recommendations
* Low-confidence diagnostic flag

---

# 10. Detection Workflow

```text
Leaf Image Upload
        ↓
FastAPI Detection Endpoint
        ↓
PyTorch Model Inference
        ↓
Confidence Threshold Validation
        ↓
Remedy Database Lookup
        ↓
Localized Response Generation
        ↓
Frontend Visualization
```

---

# 11. Localization Support

The application supports:

* English
* हिंदी (Hindi)
* ಕನ್ನಡ (Kannada)

Localized support includes:

* UI labels
* Detection outputs
* Remedy recommendations
* Advisory messaging
* Landing page content
* Homepage feature descriptions
* Navigation labels

---

# 12. Theme System

The frontend includes dual-theme support:

## Dark Mode

* Default desktop-friendly theme
* Reduced eye strain
* Modern landing page optimized for dashboard experience

## Light / Outdoor Mode

* High-contrast accessibility mode
* Improved sunlight readability for field usage
* Responsive homepage readability improvements

---

# 13. Remedy Recommendation System

The backend uses `backend/data/remedies.json` to provide structured disease treatment recommendations.

Each disease entry contains:

* Localized display names
* Organic remedies
* Chemical remedies
* Cultural farming practices

Fallback recommendations are automatically returned if a disease mapping is unavailable.

---

# 14. Confidence Threshold & Expert Workflow

The system applies a confidence threshold to identify uncertain predictions.

## High Confidence

* Standard detection results displayed
* Treatment recommendations shown normally

## Low Confidence

If prediction confidence falls below the configured threshold:

* Warning state is triggered
* Expert advisory workflow becomes available
* Users are encouraged to seek agricultural verification

This reduces the risk of unsafe treatment recommendations from uncertain predictions.

---

# 15. Model Architecture

The project uses:

* ResNet18 transfer learning
* Custom classification head
* Standardized image transforms
* Torch device-aware loading

Inference pipeline includes:

* Resize
* Tensor conversion
* Normalization
* Softmax confidence scoring

---

# 16. Development Notes

## Recommended Tools

### Linting & Formatting

```bash
uv add --dev ruff
uv run ruff check .
uv run ruff format .
```

### Type Checking

```bash
uv add --dev mypy
uv run mypy .
```

### Testing

```bash
uv add --dev pytest pytest-cov
uv run pytest --cov=. tests/
```

---

# 17. Future Improvements

Potential enhancements:

* Real-time camera detection
* Expert consultation integration
* Cloud deployment pipeline
* Additional crop support
* Offline mobile inference
* Dataset expansion
* Model explainability visualizations
* Interactive analytics dashboard on homepage
* Animated onboarding and detection walkthroughs
* Farmer education and disease awareness modules

---

# 18. License

This project is intended for educational and research purposes.

---

# 19. Acknowledgements

* PlantVillage Dataset
* PyTorch
* FastAPI
* React
* Scikit-learn
