# Project Sentinel: Hazardous Asteroid Prediction

A machine learning-powered web application that classifies Near-Earth Objects (NEOs) as hazardous or non-hazardous using asteroid physical and orbital features. The project combines data science, feature engineering, model tuning, and a user-friendly Streamlit dashboard for real-time prediction and batch analysis.

---

## Overview

This project is designed to support planetary defense analysis by predicting whether an asteroid is likely to pose a risk to Earth. It uses a trained classification model built on NASA Near-Earth Object data and exposes the workflow through an interactive dashboard.

The system supports:

- Live asteroid risk prediction with custom input parameters
- Batch CSV analysis for multiple asteroid records
- Exploratory Data Analysis (EDA) of the NEO dataset
- Model performance comparison and threshold tuning explanation
- Visualization of asteroid threat distribution and key risk indicators

---

## Problem Statement

Asteroids approaching Earth vary widely in size, speed, and miss distance. A small change in orbital parameters can dramatically affect the risk level. The goal is to identify hazardous asteroids early using historical data and machine learning patterns so that scientists and analysts can better assess potential impact threats.

---

## Project Workflow

The workflow follows a standard machine learning lifecycle, adapted for astrophysical risk analysis.

### 1. Data Collection

The project uses the NASA Near-Earth Object dataset stored in the workspace as:

- `neo__3_.csv`

This dataset includes features such as:

- absolute magnitude
- estimated diameter minimum and maximum
- relative velocity
- miss distance
- hazardous label

### 2. Data Preparation

Before model training, the dataset is cleaned and normalized for analysis:

- irrelevant columns are removed where necessary
- diameter is averaged into a single size metric
- a velocity-to-distance ratio is engineered
- missing or invalid values are handled as part of the pipeline

### 3. Feature Engineering

The model uses engineered and domain-relevant attributes to improve classification quality. Examples include:

- average asteroid diameter
- relative velocity
- miss distance from Earth
- velocity-to-distance ratio
- absolute magnitude as a proxy for size/brightness

These attributes help distinguish hazardous objects from more benign passing bodies.

### 4. Model Training

The project trains a tuned random forest classifier optimized for recall. In space-risk contexts, false negatives are more serious than false positives, so the model is tuned to detect more hazardous objects even if the precision is lower.

The final model is stored in:

- `hazardous_asteroid_model.pkl`

Supporting artifacts are also saved:

- `feature_scaler.pkl`
- `feature_columns.pkl`
- `decision_threshold.pkl`

### 5. Threshold Tuning

The decision threshold is adjusted to improve recall under a planetary defense objective.

A default threshold of `0.40` is used to balance:

- hazard detection sensitivity
- false positive control
- model recall improvement

This threshold is visible and adjustable in the Streamlit app.

### 6. Deployment as an Interactive App

The app is built with Streamlit and provides:

- live prediction form for asteroid attributes
- preset scenario buttons
- real-time probability gauge
- batch prediction for uploaded CSV files
- EDA charts and dataset exploration views
- model comparison and threshold tuning plots

---

## Application Features

### Live Asteroid Predictor

Users can manually enter values such as:

- absolute magnitude (H)
- estimated diameter min/max
- relative velocity
- miss distance from Earth

The app computes feature values, runs the model, and returns:

- hazardous or non-hazardous result
- final probability score
- impact energy estimate
- close approach distance in lunar distances

### Batch Analysis

The dashboard can:

- upload a custom CSV file
- analyze multiple asteroid records at once
- show total objects, hazardous count, safe count, and threat rate
- generate an exportable prediction registry

### Exploratory Data Analysis

The app includes visualizations for:

- class imbalance in the dataset
- asteroid magnitude distribution
- relative velocity comparison
- miss distance vs. diameter patterns

### Model Performance Panel

This section shows:

- model comparison metrics
- threshold tuning dynamics
- confusion matrix
- feature importance ranking

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

---

## Project Structure

```text
Hazardous_Asteroid_Prediction/
├── app.py                        # Streamlit dashboard application
├── hazardous_asteroid_model.pkl # trained ML model
├── feature_scaler.pkl           # feature scaler used during inference
├── feature_columns.pkl          # selected model input columns
├── decision_threshold.pkl       # tuned threshold
├── neo__3_.csv                  # NASA asteroid dataset
├── hazardous_asteroid_prediction_final.ipynb
├── NEO - ASSESMENT (3).pdf      # project report/documentation
├── README.md                    # project overview and usage guide
└── .git/                        # Git metadata
```

---

## Setup Instructions

### Prerequisites

Make sure Python is installed on your system.

### 1. Clone or Open the Project

Open the project folder in VS Code or any Python environment.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

### 4. Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## How to Use the Application

### Option 1: Live Prediction

1. Open the app.
2. Go to the "Live Asteroid Predictor" tab.
3. Enter asteroid attributes.
4. Adjust the probability threshold if needed.
5. Review the risk classification and probability gauge.

### Option 2: Batch Analysis

1. Open the "Batch Analysis & Catalog" tab.
2. Upload a CSV file or load sample NASA data.
3. Review the hazard distribution and prediction table.
4. Download the processed results as a CSV.

### Option 3: Model Insights

1. Go to the "Model Performance & Explainability" tab.
2. Study feature importance and threshold tuning.
3. Understand why the model favors specific asteroid characteristics.

---

## Model Highlights

The model is designed to prioritize asteroid detection recall, which is essential in a planetary defense context.

Key points:

- tuned random forest model
- strong ROC-AUC performance
- threshold optimized for higher hazardous object recall
- feature engineering aligned with astrophysical risk factors

---

## Important Note

This project is intended for educational, research, and prototype analysis purposes. It is not a real-time operational planetary defense system and should not be used as a sole decision-making tool for real-world hazard assessment.

---

## Future Improvements

Possible enhancements for future versions include:

- support for real NASA API integration
- additional orbital metrics such as orbit class and close approach date
- explainability tools like SHAP or LIME
- improved model comparison with XGBoost or LightGBM
- deployment to cloud hosting or public web service

---

## Credits

This project is built around the concept of asteroid risk assessment using machine learning and NASA asteroid data. It is intended to demonstrate how prediction pipelines can support planetary defense monitoring.

---

## License

This project is for educational and research use. Please check repository ownership and usage rights before redistributing or commercializing the project.
