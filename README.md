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

```
mimic-aki-mortality-prediction/
├── notebooks/
│   └── mimic_aki_pipeline.ipynb    # Full pipeline: EDA → preprocessing → modeling → evaluation
├── data/                            # Not versioned — see Data access above
├── requirements.txt
└── README.md
```

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

## Limitations

- **Single-center, single-country data.** MIMIC-III was collected at Beth Israel Deaconess
  Medical Center (Boston, USA) between 2001 and 2012. The model has not been externally
  validated on other hospitals, healthcare systems, or countries. Performance on populations
  with different demographics, comorbidity patterns, or clinical practices (e.g. European
  cohorts) is unknown and should not be assumed to transfer.
- **No temporal validation.** All data comes from a single historical period; clinical
  practice, diagnostic criteria, and case-mix in ICUs have evolved since 2012, which can
  degrade model performance over time (dataset shift).
- **Single train/test split.** While 5-fold cross-validation was used during Random Forest
  hyperparameter selection, the final reported test metrics come from one stratified 80/20
  split. Results may vary modestly with a different split (see the two Random Forest runs
  documented above, which produced slightly different optimal `n_estimators` and AUROC due
  to environment-level variation).
- **Academic scope, not a clinical decision tool.** This project has not undergone the
  validation, regulatory review, or prospective testing required for clinical use. It should
  not be compared directly to established, prospectively validated severity scores (e.g.
  APACHE II, SOFA), which have been tested across many institutions over decades. This
  repository is a methodological exercise demonstrating a complete, reproducible ML pipeline
  applied to a clinical prediction problem — not a clinically validated tool.
- **Limited variable scope.** The dataset provides 36 variables, all demographic (age,
  gender) or physiological/lab summary statistics (max/mean/min of vital signs and blood
  work) plus ICU length of stay. It does not include preexisting comorbidities (e.g.
  diabetes, chronic kidney disease, heart failure), the underlying cause of AKI (prerenal,
  renal, or postrenal — clinically important for prognosis), or specific interventions
  received (e.g. renal replacement therapy/dialysis, vasopressors, mechanical ventilation
  beyond FiO2 as an indirect proxy). The model therefore captures a simplified physiological
  snapshot of the ICU stay rather than the full clinical picture a treating physician would
  use, and its predictive performance is necessarily bounded by what these 36 variables can
  represent.
- **Global feature importance, not explainability.** The interpretability analysis in this
  project (Random Forest and XGBoost feature importance) is global and impurity/gain-based —
  it identifies which variables matter across the whole cohort, but does not explain why the
  model flagged a *specific* patient as high-risk, nor capture non-linear interactions
  between variables. XGBoost is also inherently less interpretable than Logistic Regression
  by construction (an ensemble of hundreds of trees vs. a single set of coefficients). If the
  end goal were clinical adoption, instance-level explainability methods (e.g. SHAP, LIME)
  would be a requirement this project does not yet address.

## Related work

- [NHANES periodontal health / muscle strength pipeline (PySpark, Big Data)](https://github.com/alejandrovazquez-alonso/nhanes-periodontal-muscle-strength)
- [BRFSS health risk clustering & classification](#) — link to be added

## License

MIT
