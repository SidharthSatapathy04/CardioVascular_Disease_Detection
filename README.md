#  Cardiovascular Disease (CVD) Detection

A machine learning pipeline to detect and predict the risk of cardiovascular disease using clinical features — enabling early diagnosis and preventive care.

---

##  Problem Statement

Cardiovascular disease is the leading cause of death globally. Early detection using patient data (age, blood pressure, cholesterol, etc.) can save lives. This project builds a robust ML classifier to predict CVD risk from clinical indicators with explainability built in.

---

##  Project Structure

```
CardiovascularDisease-Detection/
│
├── plots/                    # Generated visualizations (feature importance, ROC, etc.)
├── reports/                  # Evaluation reports and metrics output
│
├── preprocessing.py          # Data cleaning, encoding, normalization
├── feature_selection.py      # Select top predictive features
├── train.py                  # Model training (full run)
├── evaluate.py               # Evaluation metrics and report generation
├── explain.py                # Model explainability (SHAP / feature importance)
├── utils.py                  # Helper functions shared across scripts
│
├── main.py                   # Full pipeline runner
├── main_fast.py              # Faster pipeline (reduced dataset / quick training)
├── quick_run.py              # Minimal run for testing
│
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

##  Pipeline Overview

```
Raw Data → Preprocess → Feature Selection → Train → Evaluate → Explain → Reports & Plots
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `preprocessing.py` | Handle nulls, encode categoricals, normalize |
| 2 | `feature_selection.py` | Select top CVD-predictive features |
| 3 | `train.py` | Train classifier on selected features |
| 4 | `evaluate.py` | Accuracy, F1, AUC-ROC, confusion matrix |
| 5 | `explain.py` | SHAP values / feature importance plots |
| 6 | `plots/` | All saved visualizations |
| 7 | `reports/` | Saved evaluation reports |

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SidharthSatapathy04/CardiovascularDisease-Detection.git
cd CardiovascularDisease-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python main.py
```

### 4. Quick Test Run
```bash
python quick_run.py
```

### 5. Fast Mode (reduced training time)
```bash
python main_fast.py
```

---

##  Requirements

Key libraries used:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
xgboost
imbalanced-learn
jupyter
```

> Install all via: `pip install -r requirements.txt`

---

##  Key Features Used

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Blood Pressure | Systolic/diastolic readings |
| Cholesterol | Total cholesterol level |
| Max Heart Rate | Maximum heart rate achieved |
| Chest Pain Type | Type of chest pain (categorical) |
| Fasting Blood Sugar | Blood sugar > 120 mg/dl (boolean) |
| ST Depression | ECG reading during exercise |
| Target | `1` = CVD Present, `0` = No CVD |

---

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.7313 |
| F1 Score | 0.7205 |
| AUC-ROC | 0.8001 |
| Precision | 0.7482 |
| Recall | 0.6947 |

> Run `evaluate.py` to generate updated metrics saved in `reports/`

---

##  Explainability

`explain.py` generates SHAP-based visualizations to interpret model decisions:
- **Feature Importance Bar Chart** — which features matter most
- **SHAP Summary Plot** — direction and magnitude of each feature's impact
- All plots saved to `plots/`

---

##  Future Improvements

- [ ] Add deep learning model (MLP / TabNet)
- [ ] Build a Streamlit web app for doctor-facing predictions
- [ ] Integrate with real patient EHR data
- [ ] Add cross-validation and hyperparameter tuning logs
- [ ] Deploy as REST API (FastAPI / Flask)

---

##  Authors

**Sidharth Satapathy**
- GitHub: [@SidharthSatapathy04](https://github.com/SidharthSatapathy04)
**Biswaranjan Panda**
- GitHub: [@Biswa2006](https://github.com/Biswa2006)
---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
