# Predictive Medical Diagnostic System
### Multinomial Disease Classification from 132 Binary Symptoms · 41 Disease Categories · Advanced ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)](https://catboost.ai)

---

## Overview

An end-to-end predictive analytics pipeline that classifies **41 diseases** from **132 binary symptom features** using advanced ensemble machine learning. The core challenge is a genuinely hard problem: a **41-class multinomial target** with only 304 unique patient records (after deduplication) and severe **multicollinearity** across 132 binary features (63 highly correlated symptom pairs, |r| > 0.85).

This project was developed as part of a PhD research programme on *Implementation and Impact of Data Science using AI in Auditing and Healthcare Practices* at Mysore University.

---

## Key Technical Contributions

### 1. Multicollinearity Treatment (Two-Stage)
Standard correlation dropping is too crude for 132 interdependent binary features. This project applies a principled two-stage approach:
- **Hierarchical clustering** of features by correlation distance — groups symptom clusters (e.g., `weight_gain`, `enlarged_thyroid`, `brittle_nails` all at r > 0.93) and retains only the most informative representative per cluster, scored by Mutual Information against the target
- **Iterative VIF elimination** (threshold = 10) on the cluster survivors — removes any remaining linear dependence, producing a clean, near-orthogonal feature set

### 2. Five Dimensionality Reduction Strategies (Empirically Compared)
| Method | Features | Supervised | Best suited for |
|---|---|---|---|
| Raw binary | 132 | No | Baseline |
| PCA (95% variance) | 46 | No | Unsupervised compression |
| **LDA (40 components)** | **40** | **Yes** | **41-class multinomial ← recommended** |
| TruncatedSVD (50) | 50 | No | Sparse binary matrices |
| Hybrid (LDA + Engineered) | 62 | Mixed | Best overall performance |

LDA is theoretically optimal here: it finds projections that **maximise between-class separation** and **minimise within-class scatter** — directly aligned with the 41-class discrimination objective.

### 3. Domain-Driven Feature Engineering (10 Clinical Symptom Groups)
Rather than treating all 132 symptoms as independent signals, clinically meaningful aggregates are constructed:

| Feature | Description |
|---|---|
| `gastro_score` | Gastrointestinal symptom burden (10 symptoms) |
| `respiratory_score` | Respiratory involvement (8 symptoms) |
| `neuro_score` | Neurological symptoms (9 symptoms) |
| `cardiac_score` | Cardiovascular symptoms (6 symptoms) |
| `liver_score` | Hepatic indicators (7 symptoms) |
| `rarity_score` | Inverse-frequency weighted symptom sum — rare symptoms carry more signal |
| `symptom_entropy` | Shannon entropy across active symptoms — measures diagnostic uncertainty |
| `system_breadth` | Count of organ systems involved — distinguishes systemic from local disease |
| `dominant_system` | Encoded label of the most active organ system per patient |

### 4. Seven Ensemble Models with Intense Hyperparameter Tuning
All models use `class_weight='balanced'` — no SMOTE. RandomizedSearchCV with 40–60 iterations and Stratified 5-Fold CV throughout.

| Model | Feature Space | Key Hyperparameters Tuned |
|---|---|---|
| Random Forest | Raw + Engineered | n_estimators, max_depth, max_features, criterion, bootstrap |
| XGBoost (`multi:softmax`) | LDA (40 comp) | learning_rate, max_depth, subsample, colsample, reg_alpha, gamma |
| LightGBM (`multiclass`) | Hybrid (LDA + Eng) | num_leaves, learning_rate, subsample, reg_lambda, min_child_samples |
| CatBoost (`MultiClass`) | Raw + Engineered | depth, learning_rate, l2_leaf_reg, bagging_temperature |
| Extra Trees | Raw + Engineered | n_estimators, max_depth, max_features, criterion |
| Soft Voting (RF + LGB + Cat) | Hybrid | Combines probability estimates from 3 best models |
| Stacking (RF + ET + LGB → LR) | Raw + Engineered | Meta-learner: multinomial Logistic Regression |

---

## Pipeline Architecture

```
Raw Data (4920 × 133)
    │
    ▼
Drop duplicates → 304 × 132 clean binary features
    │
    ├─── EDA: correlation heatmap, t-SNE, PCA scree, disease profiles
    │
    ▼
Multicollinearity Treatment
    ├── Hierarchical clustering (complete linkage, dist=0.15)
    └── Iterative VIF elimination (threshold=10)
    │
    ▼
Feature Engineering
    ├── 10 clinical symptom group scores
    ├── Rarity-weighted score, entropy, system breadth
    └── Chi² + Mutual Information filter selection (top-60 union)
    │
    ▼
Dimensionality Reduction (5 strategies)
    ├── PCA · LDA · TruncatedSVD · Hybrid
    └── Empirical comparison on same model (LightGBM)
    │
    ▼
Model Training (7 models, all with class_weight='balanced')
    ├── RandomizedSearchCV (40–60 iterations per model)
    └── StratifiedKFold(5) cross-validation
    │
    ▼
Evaluation: Accuracy · F1 (Weighted + Macro) · ROC-AUC (OvR Macro) · Per-class report
```

---

## Results Summary

> Note: With only 42 test samples (1 per class), test metrics reflect individual disease predictions. Cross-validation scores provide more reliable performance estimates.

| Model | CV F1 (Weighted) | Test Accuracy | Test F1 (Weighted) | ROC-AUC |
|---|---|---|---|---|
| Random Forest | ~0.97 | ~0.95 | ~0.95 | ~0.99 |
| XGBoost (LDA) | ~0.96 | ~0.95 | ~0.95 | ~0.99 |
| LightGBM (Hybrid) | ~0.97 | ~0.95 | ~0.95 | ~0.99 |
| CatBoost | ~0.96 | ~0.93 | ~0.93 | ~0.99 |
| Stacking | ~0.97 | ~0.95 | ~0.95 | ~0.99 |

*Exact values depend on RandomizedSearchCV random state and hardware.*

---

## Repository Structure

```
medical-diagnostic-system/
├── Medical_Diagnostic_Advanced.ipynb   # Full pipeline (61 cells, 1,092 lines)
├── DiseaseTraining.csv                  # 4,920 rows × 133 cols (raw)
├── DiseaseTesting.csv                   # 42 rows × 133 cols
└── README.md
```

---

## Setup & Usage

```bash
# Clone and install
git clone https://github.com/SOHINI98/medical-diagnostic-system
cd medical-diagnostic-system
pip install numpy pandas scikit-learn xgboost lightgbm catboost statsmodels matplotlib seaborn scipy

# Run
jupyter notebook Medical_Diagnostic_Advanced.ipynb
```

**Run order inside the notebook:** execute cells top to bottom. The notebook is self-contained — data paths are relative, no external files needed beyond the two CSVs.

---

## Technologies

`Python 3.11` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `statsmodels` (VIF) · `scipy` (hierarchical clustering) · `pandas` · `numpy` · `matplotlib` · `seaborn`

---

## Author

**Sohini Sengupta**  
Senior Data Analytics Specialist, CME Group  
PhD Research Scholar — Data Analytics & AI in Auditing, Mysore University  
[LinkedIn](https://www.linkedin.com/in/sohini-sengupta-222aab18b) · [GitHub](https://github.com/SOHINI98)
