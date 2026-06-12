# Early Sepsis Prediction Using ICU Time-Series Data

### End-to-End Healthcare AI Pipeline | XGBoost | Bidirectional LSTM | Temporal Fusion Transformer | SHAP Explainability

---

## Project Summary

This project develops an end-to-end AI system for early sepsis prediction using ICU time-series data. Built independently as a third-year B.Tech student, the pipeline covers the full machine learning lifecycle from raw heterogeneous hospital data to interpretable deep learning models capable of predicting sepsis risk multiple hours in advance.

The workflow includes:

- Exploratory Data Analysis and clinical data quality assessment
- Multi-table healthcare data integration across nine ICU tables
- Temporal feature engineering with lag, rolling, and delta features
- XGBoost baseline modeling with patient-level cross-validation
- SHAP explainability analysis for clinical interpretability
- Bidirectional LSTM modeling for sequential patient trajectory learning
- Temporal Fusion Transformer for multi-horizon sepsis forecasting

Dataset statistics:

- 331,637 patient-hour records
- 2,640 ICU patients
- 97 sepsis patients
- 100 engineered features
- 47 to 1 class imbalance ratio

Best baseline performance (XGBoost, 5-fold OOF):

- AUROC: 0.9451
- AUPRC: 0.5024
- F1 Score: 0.5294
- Represents a 24x lift over a random classifier baseline

---

## Clinical Problem Statement

Sepsis kills approximately 11 million people globally every year. Every hour of delayed treatment increases mortality by 7 to 10 percent. The challenge is not treatment — effective protocols exist. The challenge is early detection.

By the time a clinician manually identifies the deterioration pattern across dozens of patients, physiological damage has already begun. ICUs generate thousands of data points per patient per day across vital signs, laboratory measurements, medications, procedures, and devices. No human can track subtle multi-feature trends in real time across an entire ward.

This system addresses that problem by:

- Continuously scoring every patient-hour for sepsis risk using all available clinical data
- Surfacing early warnings before deterioration becomes irreversible
- Explaining which specific features drove each prediction so alerts are actionable
- Forecasting risk at multiple future time horizons so clinicians receive prospective warnings, not retrospective flags

---

## Dataset Overview

The dataset consists of nine heterogeneous ICU tables linked by patient identifier and timestamped at hourly resolution.

| Table | Description | Key Variables |
|---|---|---|
| SepsisLabel | Target variable, hourly binary label | SepsisLabel (0 or 1) |
| person_demographics_episode | Patient-level static information | age_in_months, gender |
| measurement_lab | Blood tests and clinical laboratory results | Procalcitonin, creatinine, PT, lactate, WBC |
| measurement_meds | Vital signs | Heart rate, systolic BP, SpO2, temperature, respiratory rate, FiO2 |
| measurement_observation | Clinical observations | GCS, pupils, pulse pressure |
| drugsexposure | Medication administration history | Epinephrine, norepinephrine, dopamine, milrinone |
| proceduresoccurrences | Medical procedures performed | Invasive ventilation, dialysis, ECMO |
| devices | Device and equipment usage | Arterial lines, central venous catheters |
| observation | Additional clinical observation records | Supplementary clinical data |

Dataset statistics after full integration:

| Metric | Value |
|---|---|
| Total patient-hour rows | 331,637 |
| Unique patients | 2,640 |
| Sepsis patients | 97 |
| Non-sepsis to sepsis ratio | 47 to 1 |
| Final feature matrix dimensions | 331,637 rows by 100 features |
| ICU stay length range | 1 hour to 5,792 hours |

---

## Project Workflow

```
Raw ICU Data (9 tables)
         |
         v
Exploratory Data Analysis
(data quality, class imbalance, missing values, distributions)
         |
         v
Feature Engineering
(multi-table merge, temporal features, lag features, rolling statistics)
         |
         v
XGBoost Baseline
(patient-level 5-fold OOF CV, scale_pos_weight, threshold tuning)
         |
         v
SHAP Explainability
(TreeExplainer, beeswarm, dependence plots, category contribution)
         |
         v
Bidirectional LSTM
(padded sequences, Focal Loss, sequential patient modelling)
         |
         v
Temporal Fusion Transformer
(multi-horizon prediction at t+1, t+3, t+6 hours ahead)
```

---

## Notebook 00 - Exploratory Data Analysis

File: `sepsis_healthcare_eda.ipynb`

### What I did

Loaded all nine raw CSV tables and performed a systematic data quality audit before any modelling. This step is not optional in a real healthcare dataset because assumptions made here propagate through every downstream decision.

Analyses performed:

- Shape, column names, and data type inspection across all nine tables
- Missing value quantification per column and per table
- Duplicate row detection using pandas
- Duplicate file detection using MD5 hash comparison to confirm no two tables are identical copies
- Memory usage profiling per table
- Missing value matrix visualisation using missingno across all tables
- Sepsis class distribution analysis to confirm imbalance severity
- Patient-level statistics including hourly records per patient and episode lengths
- Numeric feature histograms for demographic variables
- Correlation heatmap of clinical observation features to identify collinear variables
- Datetime column identification across all tables for temporal parsing

### Critical findings

| Finding | Why It Mattered |
|---|---|
| Sepsis label at approximately 2.1 percent of all rows | 47 to 1 imbalance makes standard accuracy meaningless. AUPRC became the primary metric throughout. |
| Time-series structure at hourly resolution | Row-level train-test split would cause data leakage. Patient-level splitting became mandatory. |
| Significant missing values across multiple tables | Clinical data is inherently sparse. Forward-fill imputation strategy was required. |
| No duplicate files confirmed via MD5 hash | Data integrity confirmed before any downstream processing. |

---

## Notebook 01 - Feature Engineering

File: `01_feature_engineering_fixed__1_.ipynb`

### What I did

Built the entire feature matrix from scratch by merging all nine tables into a single patient-hour backbone and engineering clinically meaningful features across six data source categories.

### Patient-hour backbone

The sepsis label table defines the master time index. Every unique combination of person_id and measurement_datetime floor-rounded to the hour becomes one row. All other tables are left-joined onto this backbone. This ensures no patient-hour is created incorrectly and no label is lost. Sixteen rows with missing datetime values were dropped.

### Laboratory and vital sign aggregation

Multiple readings within the same hour were aggregated by mean. This is clinically appropriate because within-hour variation in a stable ICU patient is primarily measurement noise rather than physiological signal. Column names were standardised with prefixes (lab_ for laboratory, vital_ for vitals, obs_ for observations) to make feature grouping unambiguous downstream.

### Drug exposure converted to binary flags

The top 20 drugs by frequency, covering approximately 90 percent of total administration volume, were pivot-tabled into binary presence-absence flags per patient-hour. Missing hours were filled with zero.

This was a deliberate design choice over using dose values. The fact that a vasoactive drug such as norepinephrine was administered is a far stronger sepsis signal than the exact dose, because these drugs are given in response to circulatory collapse which is the defining feature of septic shock. Binary flags also eliminate dose unit inconsistencies and reduce titration noise.

### Procedures and devices converted to binary flags

The same logic applied to the top 15 procedures and top 10 devices. Invasive ventilation, dialysis, ECMO, arterial lines, and central venous catheters are all known severity markers in ICU literature. Their presence in a given hour directly indicates patient acuity level.

### Temporal feature engineering

Three categories of temporal features were engineered because XGBoost treats every row as independent and cannot see trends without explicit temporal context.

Time position features: icu_hour (hours elapsed since the patient's first ICU hour), hour_of_day, day_of_week. These capture circadian effects on physiology and ICU workflow patterns.

Lag features: Values at t-1, t-2, and t-6 for the five most clinically important vitals including heart rate, systolic blood pressure, SpO2, temperature, and respiratory rate. A rising heart rate over six hours is a far stronger signal than a single elevated reading.

Rolling window features: 3-hour and 6-hour rolling mean and standard deviation per vital. Rolling standard deviation captures physiological instability, which is clinically significant because a wildly fluctuating heart rate is more concerning than a steadily elevated but stable one.

Delta features: Change from the previous hour for each vital. This captures the direction and speed of physiological change, which is the first-order temporal derivative.

### Imputation

Forward-fill was applied within each patient's stay before computing lag and rolling features. This is clinically justified because the most recent recorded value is the best estimate of the current state when no new measurement exists.

### Final output

Feature matrix shape: 331,637 rows by 100 features. Saved as features_train.parquet. The exact column list was frozen in feature_cols.json so all downstream models use identical column ordering, preventing mismatch errors at inference.

---

## Notebook 02 - XGBoost Baseline

File: `02_xgboost_baseline__1_.ipynb`

### What I did

Trained an XGBoost classifier with patient-level 5-fold out-of-fold cross-validation to establish a strong, leak-free baseline that all subsequent deep learning models are benchmarked against.

### Why XGBoost as the baseline

XGBoost is the standard for tabular healthcare data. It handles missing values natively through its split-finding algorithm, is robust to feature scale differences, and its scale_pos_weight parameter provides a direct mechanism for imbalanced classification. Feature importance also allows rapid clinical sanity-checking before investing GPU compute on deep learning.

### Patient-level stratified cross-validation

This is the most critical methodological decision in the entire project.

Every patient's hourly records were assigned entirely to either the training fold or the validation fold. No patient appeared in both. Stratification used the patient-level ever-sepsis label, meaning whether a patient ever received a sepsis label during their stay, to maintain the prevalence ratio across folds.

Row-level splitting would have inflated AUROC by approximately 0.05 to 0.15 because validation rows would come chronologically after training rows for the same patient, leaking the patient's earlier trajectory into training. Patient-level splitting produces metrics that reflect genuine generalisation to new patients never seen during training, which is the only clinically meaningful question.

### Handling 47 to 1 class imbalance

scale_pos_weight was set to 47.0. This modifies XGBoost's loss function to penalise misclassifying a sepsis hour 47 times more than a non-sepsis hour, which is mathematically equivalent to oversampling the positive class by a factor of 47 but without introducing synthetic data or noise.

### Key hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| eval_metric | aucpr | AUPRC is the correct primary metric for severe imbalance. AUROC is misleadingly high when negatives dominate. |
| scale_pos_weight | 47.0 | Directly counteracts the 47 to 1 imbalance in the loss function. |
| min_child_weight | 10 | Forces conservative splits requiring at least 10 samples per leaf, preventing overfitting on the small sepsis group. |
| early_stopping_rounds | 50 | Halts when AUPRC on validation stops improving for 50 rounds. |
| n_estimators | 500 | Upper bound. Early stopping selects the actual optimal tree count per fold. |
| learning_rate | 0.05 | Slower learning with more trees gives better generalisation. |
| tree_method | hist | Histogram-based split finding, significantly faster on large datasets. |

### Threshold tuning

The default threshold of 0.5 is inappropriate for imbalanced classification. I scanned thresholds from 0.10 to 0.90 in steps of 0.01 and selected the value maximising F1 on each fold's validation set. This produced a global optimal threshold of 0.770 on the out-of-fold predictions.

### Results

| Metric | Score |
|---|---|
| AUROC | 0.9451 |
| AUPRC | 0.5024 |
| F1 Score | 0.5294 |
| Precision | approximately 0.52 |
| Recall | approximately 0.54 |
| Specificity | approximately 0.97 |
| Optimal threshold | 0.770 |
| Best single fold AUROC | 0.9777 (Fold 3) |

A random classifier on this dataset has AUPRC equal to approximately 0.021 (the class prevalence). The XGBoost AUPRC of 0.5024 is a 24x lift over random.

The apparent gap between AUROC of 0.9451 and AUPRC of 0.5024 is expected and not a flaw. AUROC measures rank ordering, which is relatively easy when negatives vastly outnumber positives. AUPRC measures precision at every recall level, which is genuinely hard at high recall when the positive class is rare.

### Top features by importance

Epinephrine flag, dopamine flag, norepinephrine flag, milrinone flag (all vasoactive drug flags), procalcitonin, prothrombin time, creatinine, icu_hour, and invasive ventilation flag.

---

---

## Notebook 03 - SHAP Explainability

File: `03_shap_analysis_fixed__2.ipynb`

### What I did

Applied SHAP (SHapley Additive exPlanations) using TreeExplainer on the best XGBoost fold model to produce globally and locally interpretable explanations for every prediction.

### Why SHAP over standard feature importance

XGBoost's built-in importance metrics tell you which features were used but not how they affected each individual prediction. SHAP decomposes every prediction into exact per-feature contributions that sum back to the model output. They satisfy mathematical fairness properties that standard importance measures do not: all contributions sum to the prediction, equal contributors receive equal attribution, and irrelevant features receive zero attribution.

In a clinical application this distinction matters. A clinician cannot act on "procalcitonin is an important feature." A clinician can act on "this patient's procalcitonin of 18 ng/mL increased their sepsis risk score by 0.23 log-odds." SHAP provides the second type of explanation.

### Computation

A stratified sample of 5,000 rows (1,000 sepsis and 4,000 non-sepsis) was used. TreeExplainer computes exact SHAP values for tree models, not approximations.

### Analyses performed

Global bar chart: Features ranked by mean absolute SHAP value. Confirmed vasoactive drug flags dominate, with procalcitonin as the most predictive laboratory value.

Beeswarm plot: Each point represents one patient-hour. Horizontal position is the SHAP value (positive pushes toward sepsis, negative pushes away). Colour encodes feature value (red for high, blue for low). High epinephrine flag strongly increased sepsis probability. High procalcitonin increased sepsis probability. Low SpO2 increased sepsis probability. All directions align with known clinical pathophysiology.

Dependence plots for top six features: Shows the non-linear relationship between feature value and SHAP contribution. Procalcitonin showed a sharp contribution increase above values corresponding to known clinical diagnostic thresholds, confirming the model independently learned established clinical cutoffs from data.

Feature category contribution breakdown:

| Category | Contribution Rank |
|---|---|
| Drugs (vasoactive flags) | Highest |
| Laboratory values | Second |
| Procedures | Third |
| Vital signs | Fourth |
| Temporal features | Fifth |
| Demographics | Low |
| Devices | Low |

The dominance of drug flags is clinically interpretable. Vasoactive drugs are administered in response to haemodynamic instability, meaning clinicians are already responding to deterioration before the formal sepsis label is assigned. The model learned treatment context as a proxy for physiological severity.

SHAP over ICU time: SHAP magnitude grouped into six-hour windows showed a gradual increase for sepsis patients rather than a sudden spike at diagnosis. This confirms that early warning signal exists and is detectable hours before the formal clinical diagnosis.

Local waterfall plots: Individual explanations produced for true positives, false negatives, and false positives to understand model failure modes.

---

## Notebook 04 - Bidirectional LSTM

File: `04_lstm_fixed__3_.ipynb`

### What I did

Built a PyTorch Bidirectional LSTM that treats each patient as a variable-length sequence of hourly feature vectors and predicts sepsis probability at every single timestep.

### Why LSTM after XGBoost

XGBoost processes each patient-hour as an independent row. Even with lag and rolling features, it sees a snapshot. It cannot learn that a pattern of steadily rising lactate over 12 hours combined with progressive tachycardia is more predictive than any individual value. An LSTM with its hidden state explicitly models the patient's evolving physiological trajectory across the entire stay.

### Architecture

```
Input: (batch_size, T, 100 features)

Bidirectional LSTM
   2 layers
   hidden_size = 128
   output = 256 per timestep (bidirectional doubles output)
   pack_padded_sequence for variable-length efficiency

Dropout (probability 0.3)

Linear: 256 to 1
Sigmoid activation

Output: (batch_size, T) — one probability per timestep
```

Total trainable parameters: approximately 530,000. Kept deliberately small relative to 2,640 patients and 97 sepsis cases to avoid overfitting the minority class.

### Variable-length sequence handling

ICU stays range from 1 to 5,792 hours. Padding every sequence to 5,792 timesteps wastes most of the computation budget for shorter stays. PyTorch's pack_padded_sequence runs the LSTM only over real timesteps and ignores padding. Sequences are sorted by descending length within each batch as required. Padding positions in target sequences are marked with -1 and masked from the loss.

### Focal Loss

Standard Binary Cross-Entropy on a 47 to 1 dataset causes gradients to be dominated by easy negative examples. The model quickly learns that predicting zero probability everywhere gives very low loss because 97.9 percent of timesteps genuinely are not sepsis. The result is high accuracy but zero recall on the minority class.

Focal Loss solves this through the modulating factor (1 minus p) raised to the power gamma. Well-classified easy negatives receive near-zero gradient contribution. Incorrectly classified sepsis hours receive large gradient contribution, focusing training on hard borderline cases.

Parameters: alpha equal to 0.75 (up-weighting positive class) and gamma equal to 2.0 (standard value from Lin et al. 2017).

### Preprocessing

Continuous features were z-score standardised to mean zero and standard deviation one. LSTM gate activations depend on dot products of input vectors with weight matrices. Unnormalised laboratory values in the hundreds alongside heart rate values in the tens would produce unstable training gradients. Binary drug, procedure, and device flags were left as zero and one because standardising them adds noise without benefit.

Scaler parameters saved to lstm_scaler.json for reproducible inference on new patient data.

### Training configuration

| Parameter | Value | Reason |
|---|---|---|
| Batch size | 16 | Small because variable-length sequences pad to the batch maximum. |
| Epochs | 20 | Sufficient for convergence given the dataset size. |
| Learning rate | 3e-4 | Standard Adam default for LSTMs. |
| Hidden size | 128 giving 256 bidirectional | Balances capacity against overfitting risk. |
| Dropout | 0.3 | Applied between LSTM layers and before the classifier head. |
| Folds | 5 | Same patient-level OOF split as XGBoost for direct comparability. |

---

## Notebook 05 - Temporal Fusion Transformer

File: `05_tft__1_.ipynb`

### What I did

Extended the sequential modelling approach into a simplified Temporal Fusion Transformer that simultaneously predicts sepsis risk at three future horizons: one hour ahead, three hours ahead, and six hours ahead.

### Why multi-horizon prediction

A same-hour alert has limited clinical utility because the patient is already deteriorating and interventions may already be underway. A three-hour ahead warning allows time to prepare treatments, order confirmatory tests, and escalate monitoring before the crisis arrives. Six-hour prediction extends the intervention window further. Multi-horizon forecasting is the difference between a reactive system and a genuinely proactive one.

### Why TFT after LSTM

The LSTM captures local sequential dependencies well. Predicting six hours ahead also requires integrating information from much earlier in the stay, such as trajectory trends from 18 hours ago. Multi-head self-attention over the LSTM output allows the model to learn which past timesteps are most relevant for each future horizon through learnable attention weights.

### Multi-horizon label construction

For every patient-hour at position t, three targets are created: SepsisLabel at t+1, SepsisLabel at t+3, and SepsisLabel at t+6. Positions where the future label falls outside the patient's recorded stay are marked -1 and masked from both the loss computation and all metric calculations.

### Sequence truncation

The LSTM processed full sequences up to 5,792 hours. Multi-head self-attention computes a T by T attention matrix per head. A 5,792-hour sequence produces over 33 million elements per head per sequence, which is infeasible on standard GPU. Sequences are truncated to the most recent 336 hours (14 days), keeping attention memory bounded while covering essentially all clinically relevant sepsis onset windows.

### Architecture

```
Static covariates: age, gender
   Linear projection to size 128 with ReLU

Time-varying features: (batch, T, 100)
   Bidirectional LSTM, 2 layers, hidden size 128
   Output: (batch, T, 256)

Static gate:
   Linear(128, 256) applied per timestep
   Conditions static patient context into every LSTM output

Multi-head self-attention:
   4 heads, embedding dimension 256
   Layer normalisation and residual connection

Dropout (0.3)

Three separate linear heads:
   Linear(256, 1) for t+1 horizon
   Linear(256, 1) for t+3 horizon
   Linear(256, 1) for t+6 horizon

Output: (batch, T, 3)
```

### Static covariate handling

Age and gender are patient-level constants that do not change per timestep. Feeding them through the LSTM at every timestep processes the same static information T times redundantly. They are instead encoded once per patient through a small MLP and gated into every LSTM output timestep via a learned linear transformation. This follows the core TFT design principle from Lim et al. 2020: static context should modulate temporal processing rather than repeat within it.

### Separate prediction head per horizon

A single shared head for all three horizons would force the same linear projection for predicting one hour ahead and six hours ahead. These are related but distinct tasks. The optimal features for imminent one-hour prediction likely weight acute rapidly-changing markers differently than six-hour prediction, which relies more on stable longer-term trends. Three separate heads allow the model to learn different projections from the shared attention representation.

### Expected performance pattern

Performance degrades with increasing horizon from t+1 to t+3 to t+6. This is fundamental to any forecasting system because uncertainty compounds with lead time. The degradation curve shape is itself clinically informative: a sharp drop suggests reliance on rapidly changing features, while a flat curve suggests the model has learned stable early-warning patterns.

---

## Model Results and Comparison

All models were evaluated using patient-level cross-validation to prevent data leakage between training and validation patients. Performance was measured using AUROC, AUPRC, and F1 Score, with AUPRC considered the primary metric due to the severe class imbalance (~47:1).

| Model | AUROC | AUPRC | F1 Score | Notes |
|---------|---------|---------|---------|---------|
| XGBoost Baseline (same-hour prediction) | 0.9451 | 0.5024 | 0.5294 | 5-fold OOF CV, threshold = 0.770, scale_pos_weight = 47, 24× lift over random baseline |
| Bidirectional LSTM (same-hour prediction) | 0.9481 | 0.4393 | 0.4870 | Sequential patient modelling using Focal Loss (α = 0.75, γ = 2.0), threshold = 0.520 |
| TFT (1-hour ahead prediction) | 0.9444 | 0.3337 | 0.4114 | Multi-horizon forecasting, predicts sepsis risk 1 hour before onset |
| TFT (3-hour ahead prediction) | 0.9350 | 0.3260 | 0.4023 | Multi-horizon forecasting, predicts sepsis risk 3 hours before onset |
| TFT (6-hour ahead prediction) | 0.9330 | 0.3084 | 0.3899 | Multi-horizon forecasting, predicts sepsis risk 6 hours before onset |

### Key Findings

- XGBoost achieved the highest AUPRC (0.5024) and F1 Score (0.5294), making it the strongest overall model for same-hour sepsis prediction.
- Bidirectional LSTM achieved the highest AUROC (0.9481), indicating strong ranking performance on sequential patient trajectories.
- Temporal Fusion Transformer successfully extended the problem from classification to early forecasting by predicting sepsis 1, 3, and 6 hours ahead.
- As the prediction horizon increased, performance decreased gradually, which is expected because forecasting future clinical deterioration is substantially more difficult than same-hour detection.
- All models significantly outperformed the random baseline AUPRC of approximately 0.021.

### Clinical Interpretation

The results demonstrate a trade-off between predictive accuracy and forecasting horizon.

- XGBoost is the strongest model for immediate clinical decision support.
- Bidirectional LSTM captures temporal patient dynamics and achieves strong discrimination performance.
- Temporal Fusion Transformer enables proactive risk forecasting several hours before sepsis onset, providing clinicians with additional time for intervention.

These findings suggest that gradient-boosted trees remain highly competitive on structured healthcare data, while deep sequential models provide additional value for longitudinal patient monitoring and early warning systems.

---

## Key Clinical Findings

Vasoactive drug flags were the strongest predictors across all models. Epinephrine, norepinephrine, dopamine, and milrinone ranked in the top five features by SHAP importance. These drugs are administered in response to circulatory collapse, which is a defining feature of septic shock. The model learned that treatment context reflects physiological severity because clinicians are already responding to deterioration before the formal sepsis label is applied.

Procalcitonin was the most predictive laboratory value. Procalcitonin is a validated biomarker of systemic bacterial infection. Its SHAP dependence plot showed a sharply increasing contribution above values corresponding to known clinical diagnostic thresholds, confirming the model independently learned established clinical cutoffs from the data without being told what those cutoffs were.

Coagulation and renal function markers ranked highly. Prothrombin time and creatinine are both components of the SOFA (Sequential Organ Failure Assessment) score used clinically to diagnose and stage sepsis severity. Their emergence as top features validates that the model learned from the correct clinical signals.

The sepsis risk signal builds gradually rather than appearing suddenly. SHAP magnitude over ICU time showed a progressive increase across six-hour windows for sepsis patients. This confirms that early warning is genuinely possible hours before the formal diagnosis, which is the entire premise of this system.

Multi-horizon prediction degrades with increasing lead time. Performance at t+3 and t+6 is lower than at t+1, consistent with fundamental forecasting uncertainty. The specific degradation curve provides information about whether the model has learned stable early-warning patterns or depends on rapidly changing acute markers.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 and above |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Missingno |
| Machine learning | Scikit-Learn, XGBoost |
| Deep learning | PyTorch 2.x |
| Explainability | SHAP with TreeExplainer |
| Sequence utilities | PyTorch pack_padded_sequence and pad_packed_sequence |
| Storage formats | Parquet for feature matrix, JSON for column list and scaler parameters, Pickle for model weights |
| Development environment | Google Colab with GPU runtime |

---

## Repository Structure

```
sepsis-prediction/
|
|-- notebooks/
|   |-- 00_sepsis_healthcare_eda.ipynb
|   |-- 01_feature_engineering.ipynb
|   |-- 02_xgboost_baseline.ipynb
|   |-- 03_shap_analysis.ipynb
|   |-- 04_lstm.ipynb
|   |-- 05_tft.ipynb
|
|-- outputs/
|   |-- features_train.parquet
|   |-- feature_cols.json
|   |-- oof_xgb.parquet
|   |-- xgb_best_model.pkl
|   |-- lstm_scaler.json
|   |-- confusion_matrix_xgb.png
|   |-- shap_bar.png
|   |-- shap_beeswarm.png
|   |-- shap_dependence.png
|   |-- shap_by_category.png
|
|-- README.md
|-- requirements.txt
```

---

## How to Run

Install dependencies:

```
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap torch pyarrow tqdm matplotlib seaborn missingno
```

Run notebooks in this order. Each saves artefacts consumed by the next:

```
00_sepsis_healthcare_eda.ipynb
        |
        v
01_feature_engineering.ipynb
Requires: raw CSV files in DATA_PATH
Saves: features_train.parquet, feature_cols.json
        |
        v
02_xgboost_baseline.ipynb
Requires: features_train.parquet
Saves: xgb_best_model.pkl, oof_xgb.parquet
        |
        v
03_shap_analysis.ipynb
Requires: features_train.parquet, xgb_best_model.pkl, feature_cols.json
        |
        v
04_lstm.ipynb
Requires: features_train.parquet, feature_cols.json
Saves: lstm_scaler.json, LSTM model weights
        |
        v
05_tft.ipynb
Requires: features_train.parquet, feature_cols.json
```

Update these variables in each notebook to match your data location:

```python
DATA_PATH = "/path/to/raw/training_data"
SAVE_PATH = "/path/to/outputs"
```

---

## Author

**Lasya Priya**

B.Tech Computer Science Engineering (Artificial Intelligence and Cloud)

This project focuses on early sepsis prediction using ICU time-series data and demonstrates the application of machine learning, deep learning, explainable AI, and temporal forecasting techniques in a real-world healthcare setting.

**LinkedIn:**  
https://www.linkedin.com/in/goundla-lasya-priya-55972531a/

**GitHub:**  
https://github.com/lasyapriyagoundla

**Areas of Interest:**
- Healthcare AI
- Machine Learning
- Deep Learning
- Explainable AI (XAI)
- Time-Series Forecasting
- Clinical Decision Support Systems
- Large Language Models (LLMs)
- Agentic AI Systems
- Generative AI
- AI Applications in Healthcare
