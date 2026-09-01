# Cardiovascular Disease Risk Predictor

A desktop application that predicts a patient's cardiovascular disease risk from clinical vitals, using a RandomForest classifier trained on the Kaggle Cardiovascular Disease dataset (70,000 patient records).

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project trains a RandomForest classifier on 11 clinical features (age, blood pressure, cholesterol, glucose, lifestyle factors) to predict cardiovascular disease risk, and exposes the model through a simple Tkinter GUI for interactive single-patient predictions.

## Model performance

Evaluated on a stratified 20% held-out test split (14,000 samples):

| Metric | No Disease | Disease |
|---|---|---|
| Precision | 0.71 | 0.71 |
| Recall | 0.71 | 0.71 |
| F1-score | 0.71 | 0.71 |

**Accuracy: 70.8%**

```
Confusion Matrix
                  Predicted: No    Predicted: Disease
Actual: No             4956              2032
Actual: Disease         2051              4961
```

Error rates are balanced across both classes — the model doesn't systematically over- or under-diagnose. Accuracy in the 70-73% range is consistent with published benchmarks on this dataset; the ceiling here is largely driven by label noise (self-reported cholesterol/glucose values) rather than model capacity.

## Features

- RandomForestClassifier (300 estimators) trained on 11 clinical/lifestyle features
- StandardScaler preprocessing pipeline
- Model persistence via joblib — trains once, loads the cached model on subsequent runs
- Tkinter desktop GUI for interactive predictions
- Full classification report (precision/recall/F1/confusion matrix) logged on training

## Tech stack

`Python` · `pandas` · `scikit-learn` · `joblib` · `Tkinter`

## Project structure

```
cardio-disease-predictor/
├── cdp_train.ipynb   # training pipeline + GUI app
├── requirements.txt
├── cardio_train.csv
└── README.md
```

## Setup

```bash
git clone https://github.com/siddharthhsinghh/cardio-disease-predictor.git
cd cardiovascular-disease-predictor
pip install -r requirements.txt
```

Download `cardio_train.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and place it in the project root (not committed to this repo due to size/licensing).

## Usage

```bash
python cdp_train.ipynb
```

On first run, the model trains on `cardio_train.csv` and caches itself (`cardio_model.pkl`, `scaler.pkl`, `features.pkl`). Subsequent runs load the cached model directly. The GUI then lets you enter a patient's vitals and get an instant risk prediction.

## Dataset

[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) — 70,000 records, 11 features + target label, sourced from Kaggle (via Svetlana Ulianova).

