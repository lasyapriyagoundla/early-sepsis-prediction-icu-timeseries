# Early Sepsis Prediction Using ICU Time-Series Data

## Overview

Sepsis is a life-threatening condition caused by the body's extreme response to infection. Delayed detection can lead to organ failure, septic shock, and death. Early identification of sepsis is critical because timely intervention significantly improves patient outcomes.

This project focuses on developing an AI-driven early warning system capable of identifying patients at risk of sepsis using longitudinal ICU data, including vital signs, laboratory measurements, medications, procedures, devices, and demographic information.

The objective is to transform raw ICU records into predictive features that can be used to forecast sepsis before clinical diagnosis.

---

# Problem Statement

Current sepsis diagnosis often occurs after physiological deterioration has already begun.

ICU environments continuously generate large volumes of patient data, making it difficult for clinicians to manually recognize subtle patterns associated with the onset of sepsis.

The goal of this project is to leverage machine learning and deep learning techniques to:

- Analyze ICU patient time-series data
- Identify early warning signs of sepsis
- Generate predictive risk scores
- Support early clinical decision-making

---

# Dataset

The project uses ICU patient records containing multiple healthcare data sources.

### Available Tables

| Dataset | Description |
|----------|-------------|
| Person Demographics | Age, Gender and Episode Information |
| Laboratory Measurements | Blood Tests and Clinical Lab Results |
| Clinical Measurements | Vital Signs and Medical Measurements |
| Observations | Clinical Observation Records |
| Drug Exposure | Medication Administration History |
| Procedures | Medical Procedures Performed |
| Devices | Device and Equipment Usage |
| Sepsis Labels | Target Variable |

---

# Project Workflow

```text
Raw ICU Data
        ↓
Dataset Understanding
        ↓
Exploratory Data Analysis
        ↓
Clinical Data Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Patient-Hour Dataset Creation
        ↓
Machine Learning Models
        ↓
Explainable AI
        ↓
Deep Learning Models
```

---

# Phase 1: Dataset Understanding

Completed:

- Loaded all ICU healthcare datasets
- Inspected table structures
- Identified feature types
- Analyzed dataset dimensions
- Investigated data relationships across tables

### Findings

- Multiple heterogeneous healthcare tables
- Large-scale ICU patient records
- Time-series patient monitoring data
- Significant class imbalance in sepsis labels

---

# Phase 2: Exploratory Data Analysis

Completed:

### Data Quality Assessment

- Missing value analysis
- Duplicate record detection
- Data type inspection
- Feature distribution analysis

### Target Analysis

- Sepsis vs Non-Sepsis distribution
- Patient-level statistics
- Episode-level statistics

### Findings

- Sepsis represents a minority class
- Several healthcare features contain missing values
- Time-series structure requires temporal feature extraction

---

# Phase 3: Clinical EDA

Completed:

### Clinical Feature Investigation

- Vital sign exploration
- Laboratory feature exploration
- Demographic analysis
- Correlation analysis

### Questions Explored

- Which physiological variables differ between sepsis and non-sepsis patients?
- Which clinical measurements appear most informative?
- What patient characteristics are associated with higher risk?

---

# Phase 4: Data Integration

Completed:

### Multi-Table Merging

Integrated:

- Demographics
- Laboratory Measurements
- Vital Signs
- Clinical Observations
- Drug Exposure
- Procedures
- Devices
- Sepsis Labels

Created a unified patient-level dataset for downstream modeling.

---

# Phase 5: Feature Engineering

Completed:

### Demographic Features

- Age
- Gender
- Episode Information

### Clinical Features

- Laboratory Measurements
- Vital Signs
- Observation Features

### Healthcare Activity Features

- Drug Administration Indicators
- Procedure Indicators
- Device Indicators

---

# Temporal Feature Engineering

Implemented:

### Time-Based Features

- ICU Hour
- Hour of Day
- Day of Week

### Lag Features

Previous observations were incorporated to capture historical patient behavior.

Examples:

- Previous Heart Rate
- Previous Blood Pressure
- Previous Laboratory Values

### Rolling Window Features

Generated:

- Rolling Mean
- Rolling Standard Deviation

These features help capture short-term physiological trends.

### Trend Features

Calculated:

- Change over Time
- Measurement Trends
- Temporal Deltas

---

# Data Preprocessing

Completed:

### Missing Value Handling

- Missing value assessment
- Feature-level analysis
- Imputation strategies

### Feature Cleaning

- Removed redundant features
- Removed empty columns
- Prepared final modeling dataset

---

# Baseline Machine Learning Model

Completed:

### Model

- XGBoost Classifier

### Why XGBoost?

- Strong performance on tabular healthcare data
- Handles missing values effectively
- Robust to feature interactions
- Widely used in clinical prediction tasks

### Evaluation Strategy

- Patient-level cross-validation
- Class imbalance handling
- Healthcare-focused evaluation metrics

Metrics:

- AUROC
- AUPRC
- Precision
- Recall
- F1 Score

---

# Current Status

Completed:

- Dataset Understanding
- Dataset EDA
- Clinical EDA
- Data Integration
- Feature Engineering
- Temporal Feature Creation
- Baseline XGBoost Pipeline

Current Stage:

Machine Learning Modeling

---

# Future Work

## Explainable AI

Planned:

- SHAP Analysis
- Feature Importance Analysis
- Clinical Interpretability

---

## Deep Learning Models

Planned:

### LSTM

Sequential patient modeling using historical ICU records.

### Temporal Fusion Transformer (TFT)

Advanced time-series forecasting architecture for early sepsis prediction.

---

## Multi-Horizon Sepsis Forecasting

Future objective:

Predict sepsis risk:

- 6 Hours Ahead
- 12 Hours Ahead
- 24 Hours Ahead

instead of simple binary classification.

---

# Tech Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- XGBoost

### Deep Learning (Planned)

- PyTorch
- Temporal Fusion Transformer

### Explainability (Planned)

- SHAP

---

# Repository Structure

```text
├── notebooks
│
├── 01_dataset_clinical_eda.ipynb
│
├── 02_feature_engineering_pipeline.ipynb
│
├── 03_xgboost_baseline.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# Author

Lasya Priya

B.Tech Computer Science Engineering (AI & Cloud)

Focused Areas:

- Artificial Intelligence
- Machine Learning
- Healthcare AI
- Deep Learning
- Explainable AI
