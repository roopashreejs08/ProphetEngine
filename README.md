# Classical Machine Learning for Predictive Maintenance

## Overview

This module uses classical machine learning algorithms to predict machine failures using the AI4I 2020 Predictive Maintenance Dataset. It compares Logistic Regression, Decision Tree, and Random Forest classifiers.

## Dataset

The dataset contains sensor measurements and machine-type features.

- **Training set:** 8,000 samples
- **Validation set:** 2,000 samples
- **Target variable:** `Machine failure`
- **Features:** Air temperature, process temperature, rotational speed, torque, tool wear, and machine type.

Failure-mode columns and machine identifiers were excluded from the model inputs.

## Models Implemented

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Random Forest with a tuned classification threshold

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 82.50% |
| Decision Tree | 97.80% |
| Random Forest | 97.90% |
| Random Forest (Tuned Threshold: 0.23) | 97.45% |

The default Random Forest achieved the highest validation accuracy. Threshold tuning improved failure detection recall from 42.65% to 73.53%, detecting 50 out of 68 actual failures in the validation set.

## Evaluation Metrics

The models were evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted failures that were actual failures.
- **Recall:** Proportion of actual failures detected.
- **F1-score:** Harmonic mean of precision and recall.
- **PR-AUC (Average Precision):** Measures performance across precision-recall thresholds.
- **ROC-AUC:** Measures the model's ability to rank failures above non-failures.

## Feature Importance

The Random Forest model identified the following as its most important features:

1. Rotational speed: 30.99%
2. Torque: 29.28%
3. Tool wear: 20.04%
4. Air temperature: 10.37%
5. Process temperature: 7.32%

These values represent model feature importance, not proof of causation.

## Project Structure

```text
ProphetEngine/
├── data/
│   └── processed/
│       ├── ai4i_train.csv
│       ├── ai4i_validation.csv
│       └── ai4i_metadata.json
├── src/
│   └── models/
│       ├── classical_ml.py
│       └── tune_random_forest.py
└── results/
    ├── classical_ml_metrics.csv
    ├── random_forest_feature_importance.csv
    ├── logistic_regression_confusion_matrix.png
    ├── decision_tree_confusion_matrix.png
    └── random_forest_confusion_matrix.png
```

The `results/` directory also contains the serialized trained models.

## How to Run

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

Run the model comparison:

```bash
python src/models/classical_ml.py
```

Run Random Forest threshold tuning:

```bash
python src/models/tune_random_forest.py
```

## Conclusion

The experiments show that Random Forest achieved the highest validation accuracy at 97.90% and the highest Average Precision at 0.7727. Lowering its classification threshold to 0.23 improved failure detection recall to 73.53%, with a corresponding decrease in accuracy to 97.45%.

These results are validation results. Further testing on an independent test set is recommended before making claims about generalization to unseen operational data.
