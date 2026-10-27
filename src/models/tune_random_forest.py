import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

DATA = "data/processed/"

train = pd.read_csv(DATA + "ai4i_train.csv")
valid = pd.read_csv(DATA + "ai4i_validation.csv")

target = "Machine failure"
X_train = train.drop(columns=[target])
y_train = train[target]
X_valid = valid.drop(columns=[target])
y_valid = valid[target]

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Generate out-of-fold probabilities using training data only.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_probs = cross_val_predict(
    model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)

# Select the threshold with the best F1 while requiring at least 75% recall.
p = precision[:-1]
r = recall[:-1]
f1 = 2 * p * r / (p + r + 1e-12)
eligible = np.flatnonzero(r >= 0.75)

if len(eligible) == 0:
    best_idx = int(np.argmax(f1))
    print("No threshold reached 75% training CV recall; using best F1 instead.")
else:
    best_idx = eligible[np.argmax(f1[eligible])]

threshold = float(thresholds[best_idx])
print(f"Selected threshold: {threshold:.4f}")
print(f"Training CV precision: {p[best_idx]:.4f}")
print(f"Training CV recall: {r[best_idx]:.4f}")
print(f"Training CV F1: {f1[best_idx]:.4f}")

# Fit once on all training data; evaluate on untouched validation data.
model.fit(X_train, y_train)
probs = model.predict_proba(X_valid)[:, 1]
preds = (probs >= threshold).astype(int)

print(f"\nValidation Average Precision: {average_precision_score(y_valid, probs):.4f}")
print("Validation confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_valid, preds, labels=[0, 1]))
print("\nValidation classification report:")
print(classification_report(y_valid, preds, zero_division=0))

# Feature importance for the report.
importance = pd.Series(
    model.feature_importances_, index=X_train.columns
).sort_values(ascending=False)

print("\nRandom Forest feature importance:")
print(importance.to_string())
importance.to_csv("results/random_forest_feature_importance.csv")
