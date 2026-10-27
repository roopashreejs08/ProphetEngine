from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

DATA = Path("data/processed")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "ai4i_train.csv")
valid = pd.read_csv(DATA / "ai4i_validation.csv")

target = "Machine failure"
X_train = train.drop(columns=[target])
y_train = train[target]
X_valid = valid.drop(columns=[target])
y_valid = valid[target]

# Prevent accidental use of target-related or failure-mode columns.
assert list(X_train.columns) == list(X_valid.columns)
assert not any(
    col in X_train.columns
    for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

rows = []

for name, model in models.items():
    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_valid, predictions),
        "Precision": precision_score(y_valid, predictions, zero_division=0),
        "Recall": recall_score(y_valid, predictions, zero_division=0),
        "F1-score": f1_score(y_valid, predictions, zero_division=0),
        "Average Precision (PR-AUC)": average_precision_score(
            y_valid, probabilities
        ),
        "ROC-AUC": roc_auc_score(y_valid, probabilities),
    }
    rows.append(scores)

    for metric, value in scores.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_valid, predictions, zero_division=0))

    cm = confusion_matrix(y_valid, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No failure", "Failure"],
    )
    display.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_")
    plt.savefig(OUT / f"{filename}_confusion_matrix.png", dpi=150)
    plt.close()

    # Save each trained model.
    import joblib
    joblib.dump(model, OUT / f"{filename}.joblib")

results = pd.DataFrame(rows).sort_values(
    "Average Precision (PR-AUC)", ascending=False
)
results.to_csv(OUT / "classical_ml_metrics.csv", index=False)

print("\n\nFINAL COMPARISON")
print(results.round(4).to_string(index=False))
print("\nSaved metrics to results/classical_ml_metrics.csv")
print("Saved confusion matrices and trained models in results/")
