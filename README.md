# In-Hospital Mortality Prediction in ICU Patients with Acute Kidney Injury (MIMIC-III)

Machine learning pipeline to predict in-hospital mortality (IHM) in ICU patients with Acute
Kidney Injury (AKI, KDIGO criteria), using clinical data derived from **MIMIC-III**.

**Author:** Alejandro Vázquez Alonso — Physiotherapist (9+ years clinical experience, ES/FR)
transitioning into Data Science / AI Engineering in healthtech. Master's in AI and Big Data
in Health, Universitat Autònoma de Barcelona (UAB), 2025-2027.

## Project origin

This pipeline was originally developed as part of a group assignment (4 members) for Module 2
of the Master's program. This repository is an **individual reconstruction for portfolio
purposes**, covering the preprocessing, modeling, threshold selection and interpretability
work developed personally for that assignment. It does not reproduce the group's submitted
notebook or any teammate's protected work.

## Data access

MIMIC-III is a credentialed clinical database (PhysioNet, requires CITI training completion
and a signed Data Use Agreement). **No patient data is included in this repository.**
To reproduce the pipeline, obtain your own authorized access at
[physionet.org/content/mimiciii](https://physionet.org/content/mimiciii/) and place the
resulting CSV as an input dataset (`data/ihm_aki.csv` locally, or `/kaggle/input/<dataset>/AKI.csv`
if running on Kaggle).

## Problem

- **Target:** IHM (in-hospital mortality), binary — 1 = deceased, 0 = survived
- **Cohort:** 3,550 ICU patients with AKI
- **Class imbalance:** ~27.4% mortality rate (ratio ~2.6:1)

## Pipeline overview

| Stage | Approach |
|---|---|
| Data cleaning | Physiologically implausible values converted to NaN before imputation |
| Missing values | Median imputation, fit on train only |
| Train/test split | 80/20 stratified, performed **before** imputation and scaling to avoid leakage |
| Class imbalance | `class_weight='balanced'` (LR, RF) and `scale_pos_weight` (XGBoost) instead of SMOTE — avoids synthetic clinically-implausible records in a moderate-size dataset |
| Models compared | Logistic Regression (L2, baseline), Random Forest (n_estimators selected via elbow test + 5-fold CV), XGBoost |
| Threshold selection | Weighted Youden Index (3× sensitivity weight) — reflects the higher clinical cost of false negatives in ICU mortality prediction |
| Evaluation | AUROC, F1, sensitivity, specificity, balanced accuracy; explicit train-vs-test overfitting analysis |
| Interpretability | Feature importance comparison between Random Forest and XGBoost, cross-validated against clinical plausibility |

## Key findings (actual run, executed on Kaggle)

| Model | AUROC train | AUROC test | Train-test gap |
|---|---|---|---|
| Logistic Regression (L2) | 0.8892 | 0.8724 | +0.017 (no significant overfitting) |
| Random Forest (300 trees, CV5-selected) | 0.9893 | 0.8793 | +0.110 (apparent overfitting) |
| **XGBoost** | 0.9997 | **0.8946** | +0.105 (apparent overfitting) |

- **XGBoost selected as final model** — best test AUROC (0.8946).
- Threshold selection was done in two passes: a first pass using a weighted Youden Index
  (3× sensitivity) produced τ≈0.146 (90% sensitivity / 76% specificity); a second, refined
  pass selected the threshold maximizing F1 per model and ranked models by mean(AUROC, F1) —
  since the assignment evaluates both metrics — yielding **τ = 0.26** for XGBoost
  (F1 = 0.710, sensitivity = 78.0%, specificity = 84.3%, 43 false negatives out of 195
  deaths in the test set).
- Cohort: 3,550 patients, 974 deaths (27.4% mortality, ratio 2.64:1). Train 2,840 / test 710,
  stratified split.
- Class weights (train): survivors = 0.69, deceased = 1.82 — a missed death weighs ~2.6×
  more than a missed survival during training.
- 9 variables shared in the top-10 feature importance of both Random Forest and XGBoost:
  `age`, `bic_mean`, `bp_mean`, `bp_min`, `bun_max`, `bun_mean`, `gcs_max`, `gcs_mean`,
  `gcs_min` — consistent with established ICU severity and renal function markers.

## Repository structure
## Requirements

```bash
pip install -r requirements.txt
```

## Author's note on AI assistance

This pipeline was developed with the assistance of generative AI (Claude, Anthropic) for code
optimization, conceptual explanation of methodological decisions, and clinical interpretation
validation. All final methodological design decisions and clinical interpretation were
reviewed and validated by the author, drawing on 9+ years of clinical experience as a
physiotherapist.

## Related work

- [NHANES periodontal health / muscle strength pipeline (PySpark, Big Data)](#) — link to be added
- [BRFSS health risk clustering & classification](#) — link to be added
- [BPPV clinical decision support tool (React)](#) — link to be added

## License

MIT
