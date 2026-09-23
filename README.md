# Baseline Predictive Pipeline -- ETAI
# Marta Romão 2023-1834

## Introduction

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. 
See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## week 2

*Logistic Regression*
  Train accuracy: 0.679
  Test accuracy:  0.678
  Gap (train - test): +0.001

Train and test accuracy are nearly identical 0.679 and 0.678 having a gap realy near 0 ,
showing the model generalizes well with no overfitting.

*Decision Tree*
  Train accuracy: 0.829
  Test accuracy:  0.633
  Gap (train - test): +0.196

Train accuracy 0.829 is much higher than test accuracy 0.633 having a gap of +0.196 a clear sign of overfitting. 
The tree memorized the training data instead of learning generalizable patterns.

**Overall
Logistic regression is the more reliable model here. Although its training accuracy is lower, it performs consistently on unseen data. 
The decision tree looks stronger on paper , but that advantage doesn't hold on test data, making it less trustworthy for real predictions.

## week 3
EDA- ver missing values
  - ver valores que não fazem sentido (?, -)
  - ver duplicados
  - ver multicolinearidade

Preprocessing - standardizar nome das variaveis
              - mudar data types
              - remover valores que não fazem sentidos, extremos

*Logistic Tree*
Train accuracy: 0.676
Test accuracy:  0.657
Gap (train - test): +0.019

This shows that the model generalizes reasonably well, with only a small difference between training and test performance 
and no strong signs of overfitting.

*Decision Tree*
Train accuracy: 0.679
Test accuracy:  0.678
Gap (train - test): +0.001

This suggests that the model generalizes very well to unseen data, with no clear signs of overfitting.

**Overall
The Decision Tree performs slightly better on the test data, with an accuracy of 0.678 compared to 0.657 for Logistic Regression. It also has a much smaller train-test gap, suggesting more consistent performance between training and unseen data.

Compared to Week 2, the Decision Tree improved significantly on test data, while the difference between training and test accuracy became almost zero. Logistic Regression also remains relatively stable, with only a small gap between train and test performance.

## RUN

**Windows**
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```powershell
venv\Scripts\activate
python main.py
```
## push
git add .
git commit -m "comment"
git push