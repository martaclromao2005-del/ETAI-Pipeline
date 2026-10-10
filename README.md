# Baseline Predictive Pipeline -- ETAI
# Marta Romão 20231834

## Introduction

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. 


## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py               # model construction + build_pipeline(): preprocessing + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk
├── results/                  # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |

## week 2

| **Logistic Regression** | |
|---|---:|
|Train accuracy| 0.679|
|Test accuracy|  0.678|
|Gap (train - test)| +0.001|

Train and test accuracy are nearly identical 0.679 and 0.678 having a gap realy near 0 ,
showing the model generalizes well with no overfitting.

| **Decision Tree** | |
|---|---:|
|Train accuracy| 0.829|
|Test accuracy|  0.633|
| Gap (train - test)| +0.196|

Train accuracy 0.829 is much higher than test accuracy 0.633 having a gap of +0.196 a clear sign of overfitting. 
The tree memorized the training data instead of learning generalizable patterns.

**Overall**
Logistic regression is the more reliable model here. Although its training accuracy is lower, it performs consistently on unseen data. 
The decision tree looks stronger on paper , but that advantage doesn't hold on test data, making it less trustworthy for real predictions.

## week 3
**EDA**
  - ver missing values
  - ver valores que não fazem sentido (?, -)
  - ver duplicados
  - ver multicolinearidade

**Preprocessing** 
- standardizar nome das variaveis
- mudar data types
- remover valores que não fazem sentidos, extremos

| **Logistic Regression** | |
|---|---:|
|Train accuracy| 0.676|
|Test accuracy|  0.657|
|Gap (train - test)| +0.019|

This shows that the model generalizes reasonably well, with only a small difference between training and test performance 
and no strong signs of overfitting.

| **Decision Tree** | |
|---|---:|
|Train accuracy| 0.679|
|Test accuracy|  0.678|
|Gap (train - test)| +0.001|

This suggests that the model generalizes very well to unseen data, with no clear signs of overfitting.

**Overall**
The Decision Tree performs slightly better on the test data, with an accuracy of 0.678 compared to 0.657 for Logistic Regression. It also has a much smaller train-test gap, suggesting more consistent performance between training and unseen data.

Compared to Week 2, the Decision Tree improved significantly on test data, while the difference between training and test accuracy became almost zero. Logistic Regression also remains relatively stable, with only a small gap between train and test performance.

## Week 4

**Cross Validation**

| **Logistic Regression** | |
|---|---:|
|Train accuracy| 0.675 |
|Test accuracy|   0.673|
|Gap (train - test)| +0.002|

This shows that the model generalizes very well, with almost no difference between training and test performance and no clear signs of overfitting.

| **Decision Tree** | |
|---|---:|
|Train accuracy| 0.801|
|Test accuracy|  0.625|
|Gap (train - test)| +0.176|

This suggests that the model performs much better on the training data than on unseen data, indicating a strong sign of overfitting.

| **Random Forest** | |
|---|---:|
| Train accuracy | 0.801 |
| Test accuracy | 0.631 |
| Gap (train - test) | +0.170 |

This also shows a considerable difference between training and test performance, suggesting that the model is overfitting the training data and does not generalize as well to unseen data.

**Overall**
The Logistic Tree performs best on the test data, with an accuracy of 0.673 and the smallest train-test gap (+0.002). Compared to Week 3, its test accuracy improved slightly, while the Decision Tree and Random Forest show stronger signs of overfitting.

## Week 5

The pipeline uses Optuna to try 30 settings per model. Each setting is scored with 5-fold CV on the development set only, so the test set is never used. Nested CV gives the honest score.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | `C=0.072` |
| Random forest | 0.647 ± 0.016 | 0.679 ± 0.014 | 0.682 | +0.002 | `n_estimators=124`, `max_depth=5`, `min_samples_leaf=11`, `max_features=log2` |

Tuning helped the decision tree a lot (+0.062) and the random forest a bit (+0.032), both more than the fold-to-fold std. Logistic regression did not improve (+0.001). After tuning, the three models are almost tied (0.672 to 0.679), so tuning did not clearly change the best model. I would report the random forest's nested CV score (0.679), not the best-trial score (0.682), because the best trial is the luckiest result and a bit too optimistic.

**Overall**
Tuning fixed the overfitting. The decision tree went from 0.625 (Week 4) to 0.672, and the random forest from 0.631 to 0.679. Logistic regression stayed at 0.673 because it has little to tune. The train-validation gaps are now small (about +0.01 to +0.02), so the models generalise much better. The three models are now almost tied (0.672 to 0.679), and the random forest is only nominally ahead. Train accuracy looks lower than before, but that is a good sign: the models stopped memorising. 


## RUN

```Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
powershell
venv\Scripts\activate
python main.py
```
## PUSH
```git add .
git commit -m "comment"
git push
```