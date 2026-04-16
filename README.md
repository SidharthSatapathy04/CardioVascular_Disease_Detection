# Cardiovascular Disease Prediction

This project builds and evaluates machine learning models to predict cardiovascular disease from patient health records. It includes a full preprocessing pipeline, multiple training workflows, evaluation utilities, explainability with SHAP, feature importance comparison, and a notebook for interactive analysis.

## Project Goals

- Predict whether a patient is likely to have cardiovascular disease.
- Compare multiple machine learning models on the same processed dataset.
- Generate visual reports and saved model artifacts.
- Add explainability using SHAP to understand feature impact.

## Dataset

The project uses `data/cardio_train.csv`.

- Records: `70,000`
- Columns: `13`
- Target column: `cardio`
- Class distribution:
  - `0`: 35,021
  - `1`: 34,979

Main raw columns:

`id`, `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`, `cardio`

## Features and Preprocessing

The preprocessing pipeline is implemented in `preprocessing.py` and includes:

1. Data loading with automatic delimiter fallback (`;` or `,`)
2. Missing value handling using mean imputation for numeric columns
3. Duplicate row removal
4. Outlier removal on `ap_hi` and `ap_lo` using the IQR rule
5. Feature engineering
6. Stratified train-test split
7. Standardization with `StandardScaler`
8. Class balancing with `SMOTE`

### Engineered Features

The pipeline creates these additional features:

- `bmi`
- `pulse_pressure`
- `mean_arterial_pressure`
- `lifestyle_risk_score`
- `health_risk_score`
- `age_bmi_interaction`
- `bp_category`

Note: `age` is first converted from days to years during feature engineering.

## Models

Depending on the entry point, the project can train a subset or a larger set of models.

### Full Pipeline (`main.py`)

- Logistic Regression
- Random Forest
- XGBoost
- SVM
- Neural Network
- Stacking Ensemble

### Fast Pipeline (`main_fast.py`)

- Logistic Regression
- Random Forest
- XGBoost
- SVM

### Ultra-Fast Pipeline (`quick_run.py`)

- Logistic Regression
- Random Forest
- XGBoost

## Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Generated evaluation plots include:

- Correlation heatmap
- Feature distributions
- Confusion matrices
- Combined ROC curves
- Model comparison chart
- Neural network training history
- SHAP summary plot
- SHAP importance bar chart

## Explainability

Explainability is implemented in `explain.py`.

- Tree models use `shap.TreeExplainer`
- Logistic Regression uses `shap.LinearExplainer`
- Other models fall back to `shap.KernelExplainer`

The project supports:

- Global SHAP summary plots
- SHAP bar importance plots
- Local waterfall plots for individual predictions
- Ranked top features by mean absolute SHAP value

## Current Saved Results

The current `reports/model_results.csv` contains these evaluation scores:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7259 | 0.7545 | 0.6672 | 0.7082 | 0.7918 |
| Random Forest | 0.7329 | 0.7588 | 0.6804 | 0.7174 | 0.7995 |
| XGBoost | 0.7320 | 0.7510 | 0.6918 | 0.7201 | 0.7986 |

Based on the saved report, `XGBoost` currently has the best F1-score.

## Project Structure

```text
CVD_FINAL/
|-- data/
|   `-- cardio_train.csv
|-- models/
|   |-- logistic_regression.pkl
|   |-- random_forest.pkl
|   `-- xgboost.pkl
|-- plots/
|   |-- confusion_matrices.png
|   |-- correlation_heatmap.png
|   |-- feature_distributions.png
|   |-- model_comparison.png
|   |-- nn_training_history.png
|   |-- roc_curves_combined.png
|   |-- shap_importance.png
|   `-- shap_summary.png
|-- reports/
|   |-- model_results.csv
|   `-- project_summary.txt
|-- CVD_project.ipynb
|-- evaluate.py
|-- explain.py
|-- feature_selection.py
|-- main.py
|-- main_fast.py
|-- preprocessing.py
|-- quick_run.py
|-- requirements.txt
|-- train.py
|-- utils.py
`-- README.md
```

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### Full workflow

```bash
python main.py
```

### Faster workflow

```bash
python main_fast.py
```

### Quickest workflow

```bash
python quick_run.py
```

### Notebook

Open `CVD_project.ipynb` in Jupyter Notebook or VS Code for interactive exploration.

## Key Modules

- `preprocessing.py`: data cleaning, feature engineering, scaling, SMOTE
- `train.py`: model training, cross-validation, feature importance extraction
- `evaluate.py`: metrics, ROC curves, confusion matrices, model comparison plots
- `explain.py`: SHAP explainability utilities
- `feature_selection.py`: chi-square ranking and feature importance comparison
- `utils.py`: saving models, plots, reports, and helper functions

## Outputs

Running the project generates:

- Trained models in `models/`
- Visualizations in `plots/`
- Evaluation metrics in `reports/model_results.csv`
- Summary report in `reports/project_summary.txt`

## Requirements

Main libraries used in this project:

- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- tensorflow
- shap
- matplotlib
- seaborn
- joblib
- scipy

## Notes

- `main.py` is the most comprehensive pipeline.
- `main_fast.py` is a practical balance between speed and model coverage.
- `quick_run.py` is best when you want results quickly.
- The repository already contains generated models, plots, and reports from a previous run.

## Future Improvements

- Add command-line arguments for dataset path and model selection
- Save preprocessing objects such as the scaler for deployment
- Add unit tests for preprocessing and evaluation logic
- Export prediction APIs for real-world use
- Improve the generated summary report formatting and consistency

## Disclaimer

This project is for educational and machine learning experimentation purposes. It should not be used as a substitute for professional medical diagnosis or clinical decision-making.
