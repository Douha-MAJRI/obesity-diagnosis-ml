# 🧬 Obesity Diagnosis — ML Pipeline & Streamlit Deployment

> **End-to-end medical ML project** · Python · Scikit-learn · Streamlit · Docker · CI/CD  
> Coding Week — École Centrale Casablanca · March 2025

---

## 🎯 Objective

Build a medical decision-support application to predict a patient's obesity level (7 classes) from physical and behavioral data — comparing 4 ML models, handling class imbalance, and deploying a doctor-facing Streamlit interface with SHAP explanations.

---

## 📊 Results

| Model | Dataset | Balancing | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ✅ | Filtered | Over + Under | **0.95** | **0.88** | **0.9985** |
| LightGBM | Filtered | Oversampling | 0.9574 | 0.80 | 0.9983 |
| CatBoost | Filtered | Undersampling | 0.9598 | 0.75 | 0.9981 |
| XGBoost | Approached | Class Weights | 0.9550 | 0.84 | 0.9968 |

**Selection criterion:** F1-Score (most relevant for imbalanced classes) + SHAP coherence (medically interpretable features: weight, BMI, eating habits).

---

## 🔬 Pipeline

### 1. Data Processing (`data/`)
- Dataset: `ObesityDataSet_raw_and_data_sinthetic.csv` — 2,000+ observations, 7 obesity classes
- Missing values: none detected
- One-Hot Encoding of categorical variables
- Normalization of continuous variables
- **Outlier handling:** 2 datasets produced — winsorizing vs. removal — for comparative evaluation
- **Class imbalance (7 slightly imbalanced classes):** 4 strategies tested:
  - Oversampling (SMOTE)
  - Undersampling
  - SMOTE + Undersampling combination
  - Class Weights

### 2. Model Training (`model/`)
Each model trained on both datasets × each balancing method → **32 configurations evaluated**.  
Final comparison on: Accuracy, F1-Score, ROC-AUC, Confusion Matrix, SHAP graph.

### 3. Application (`view/`)
- **Streamlit** interface for doctors: real-time prediction + SHAP feature importance
- "Feature Legend" bar (on demand)
- "Understanding the SHAP graph" mode (on demand)
- **Doctor feedback** feature: corrections collected to a document for future fine-tuning

---

## 🏗️ Architecture (MVC)

```
obesity-diagnosis-ml/
├── model/
│   ├── train_model.py          # Model training & saving
│   ├── evaluate_model.py       # Evaluation & metrics
│   ├── explain_model.py        # SHAP-based explanation
│   └── obesity_model.pkl       # Trained Random Forest
├── controller/
│   ├── predict.py              # Prediction logic
│   └── feedback_handler.py     # Doctor feedback recording
├── view/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned dataset (dataset.csv)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_data.py
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions CI/CD
└── requirements.txt
```

---

## 🛠️ Stack

| Category | Tools |
|---|---|
| ML | Scikit-learn · XGBoost · LightGBM · CatBoost |
| Imbalanced learning | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Data | Pandas · NumPy |
| UI | Streamlit |
| Deployment | Docker · docker-compose |
| CI/CD | GitHub Actions |

---

## 🚀 Run locally

```bash
# Clone
git clone https://github.com/Douha-MAJRI/obesity-diagnosis-ml
cd obesity-diagnosis-ml

# Install dependencies
pip install -r requirements.txt

# Train model
python model/train_model.py

# Launch app
streamlit run view/app.py
```

**Or with Docker:**

```bash
docker-compose up --build
# App available at http://localhost:8501
```

---

## 📈 Next improvements

- REST API (FastAPI) for integration with other medical tools
- Neural network model to improve accuracy on ambiguous classes
- Intelligent feedback loop: dynamically retrain on collected doctor corrections

---

## 👥 Team

Douha MAJRI · Ilyas BAJJA · Hajar NAJIB · Meriem LAAROUSSI · Hiba CHUIMI  
*École Centrale Casablanca — Coding Week, March 2025*
