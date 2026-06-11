# Early Sepsis Prediction Using ICU Time-Series Data

## Overview

Sepsis is a life-threatening medical condition caused by the body's extreme response to infection. Early detection can significantly reduce mortality and improve patient outcomes.

This project focuses on analyzing ICU patient time-series data to identify patterns associated with the onset of sepsis and build an AI-powered early warning system.

---

## Objectives

- Understand ICU healthcare datasets
- Perform Dataset EDA
- Perform Clinical EDA
- Identify physiological indicators associated with sepsis
- Develop machine learning and deep learning models for early prediction
- Build explainable and clinically interpretable AI solutions

---

## Dataset

Dataset Source:

PhysioNet Sepsis Challenge Dataset

Data includes:

- Patient demographics
- Vital signs
- Laboratory measurements
- Medications
- Procedures
- Device records
- Sepsis labels

---

## Exploratory Data Analysis

Completed:

- Dataset overview
- Missing value analysis
- Duplicate analysis
- Class imbalance analysis
- Patient-level statistics
- Healthcare feature exploration

---

## Key Findings

- Sepsis cases represent a minority class, resulting in significant class imbalance.
- Several clinical measurements contain substantial missing values.
- Patient records exhibit longitudinal time-series behavior.
- Healthcare features show varying relationships with sepsis occurrence.

---

## Future Work

- Feature engineering
- Patient timeline construction
- XGBoost baseline model
- LSTM-based prediction
- Temporal Fusion Transformer (TFT)
- Explainable AI using SHAP
- Multi-horizon sepsis forecasting

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- PyTorch

---

## Project Status

Current Phase:

Dataset EDA and Clinical EDA
