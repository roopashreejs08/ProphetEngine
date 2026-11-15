"""Model scoring and evaluation helpers (pure functions, no Streamlit).

A machine is flagged as "at risk" when its predicted failure probability is
at or above the decision threshold (``p >= threshold``), the same rule the
threshold-tuning script used.
"""

import math

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from dashboard import config as C


def predict_proba(model, features):
    """Failure probability for each row, using the training-time column order."""
    return np.asarray(model.predict_proba(features[C.FEATURE_COLUMNS])[:, 1], dtype=float)


def metrics_at_threshold(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape != probabilities.shape or y_true.size == 0:
        raise ValueError("y_true and probabilities must be non-empty and the same length.")
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / y_true.size),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "flagged": int(tp + fp),
        "failures": int(tp + fn),
    }


def threshold_sweep(y_true, probabilities, thresholds=None):
    """Precision / recall / F1 / alert counts across a grid of thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []
    positives = int(y_true.sum())
    for threshold in thresholds:
        flagged = probabilities >= threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = positives - tp
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / positives if positives else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (tp and precision + recall) else 0.0
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall,
                     "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def add_costs(sweep, cost_missed, cost_false_alarm):
    """Total cost of missed failures plus false alarms at each threshold."""
    out = sweep.copy()
    out["cost"] = out["fn"] * float(cost_missed) + out["fp"] * float(cost_false_alarm)
    return out


def best_threshold(sweep, column="cost", minimise=True):
    idx = sweep[column].idxmin() if minimise else sweep[column].idxmax()
    return sweep.loc[idx]


def curves(y_true, probabilities):
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return {"precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr}


def reliability(y_true, probabilities, bins=10):
    """Observed failure rate per predicted-probability bin (quantile bins)."""
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"predicted": predicted, "observed": observed})


def risk_tier(probability, threshold):
    """Critical at/above threshold, Watch from half the threshold, else Healthy."""
    if probability is None or (isinstance(probability, float) and math.isnan(probability)):
        return "unknown"
    if probability >= threshold:
        return "critical"
    if probability >= threshold / 2:
        return "watch"
    return "healthy"


def risk_tiers(probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=float)
    return np.where(probabilities >= threshold, "critical",
                    np.where(probabilities >= threshold / 2, "watch", "healthy"))


def recall_by_failure_mode(frame_with_modes, probabilities, threshold):
    """How many failures of each documented type the model flags."""
    flagged = np.asarray(probabilities) >= threshold
    failed = frame_with_modes[C.TARGET].to_numpy() == 1
    rows = []
    for code, name in C.FAILURE_MODES.items():
        mask = (frame_with_modes[code].to_numpy() == 1) & failed
        total = int(mask.sum())
        if total == 0:
            continue
        caught = int((flagged & mask).sum())
        rows.append({"code": code, "mode": name, "failures": total, "caught": caught, "recall": caught / total})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Domain reference: the documented rules used to generate AI4I failures.
# Shown next to predictions as context; the model never sees these flags.
# ---------------------------------------------------------------------------
OSF_LIMITS = {"L": 11_000, "M": 12_000, "H": 13_000}


def derived_quantities(values, machine_type):
    rpm = float(values["Rotational speed [rpm]"])
    torque = float(values["Torque [Nm]"])
    wear = float(values["Tool wear [min]"])
    return {
        "power_w": torque * rpm * 2 * math.pi / 60.0,
        "temp_diff_k": float(values["Process temperature [K]"]) - float(values["Air temperature [K]"]),
        "strain_min_nm": wear * torque,
        "osf_limit": OSF_LIMITS[machine_type],
    }


def physics_checks(values, machine_type):
    """Evaluate the AI4I dataset's documented failure conditions for one reading."""
    d = derived_quantities(values, machine_type)
    rpm = float(values["Rotational speed [rpm]"])
    wear = float(values["Tool wear [min]"])
    return [
        {
            "code": "HDF",
            "name": "Heat dissipation",
            "rule": "temperature gap < 8.6 K and speed < 1,380 rpm",
            "value": f"gap {d['temp_diff_k']:.1f} K, {rpm:,.0f} rpm",
            "triggered": d["temp_diff_k"] < 8.6 and rpm < 1380,
        },
        {
            "code": "PWF",
            "name": "Power",
            "rule": "power outside 3,500 - 9,000 W",
            "value": f"{d['power_w']:,.0f} W",
            "triggered": d["power_w"] < 3500 or d["power_w"] > 9000,
        },
        {
            "code": "OSF",
            "name": "Overstrain",
            "rule": f"tool wear x torque > {d['osf_limit']:,} min*Nm (type {machine_type})",
            "value": f"{d['strain_min_nm']:,.0f} min*Nm",
            "triggered": d["strain_min_nm"] > d["osf_limit"],
        },
        {
            "code": "TWF",
            "name": "Tool wear",
            "rule": "tool replaced or fails between 200 and 240 min",
            "value": f"{wear:,.0f} min",
            "triggered": 200 <= wear <= 240,
        },
    ]


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def sensitivity(model, values, machine_type, column, grid):
    """Failure probability as one sensor sweeps across ``grid``, others held fixed."""
    base = {c: float(values[c]) for c in C.SENSOR_COLUMNS}
    rows = []
    for value in grid:
        row = dict(base)
        row[column] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for kind in C.MACHINE_TYPES:
        frame[f"Type_{kind}"] = int(machine_type == kind)
    return predict_proba(model, frame[C.FEATURE_COLUMNS])


def local_contributions(model, values, machine_type, reference):
    """Change in probability when each sensor is reset to its training median.

    A simple, model-agnostic "what if this reading were typical?" attribution:
    a positive number means the current value pushes risk up.
    """
    current = predict_proba(model, _row(values, machine_type))[0]
    rows = []
    for column in C.SENSOR_COLUMNS:
        typical = dict(values)
        typical[column] = reference[column]["median"]
        prob = predict_proba(model, _row(typical, machine_type))[0]
        rows.append({"feature": C.SENSOR_LABEL[column], "column": column, "contribution": current - prob})
    return current, pd.DataFrame(rows)


def _row(values, machine_type):
    row = {c: float(values[c]) for c in C.SENSOR_COLUMNS}
    for kind in C.MACHINE_TYPES:
        row[f"Type_{kind}"] = int(machine_type == kind)
    return pd.DataFrame([row], columns=C.FEATURE_COLUMNS)


def partial_dependence(model, frame, column, grid):
    """Average predicted probability over ``frame`` with ``column`` fixed to each grid value."""
    features = frame[C.FEATURE_COLUMNS].copy()
    out = []
    for value in grid:
        features[column] = float(value)
        out.append(float(predict_proba(model, features).mean()))
    return np.asarray(out)


def probability_surface(model, x_col, x_grid, y_col, y_grid, fixed, machine_type):
    """Probability on a 2-D grid of two sensors, other inputs fixed."""
    xx, yy = np.meshgrid(x_grid, y_grid)
    frame = pd.DataFrame({c: np.full(xx.size, float(fixed[c])) for c in C.SENSOR_COLUMNS})
    frame[x_col] = xx.ravel()
    frame[y_col] = yy.ravel()
    for kind in C.MACHINE_TYPES:
        frame[f"Type_{kind}"] = int(machine_type == kind)
    return predict_proba(model, frame).reshape(xx.shape)


def permutation_drop(model, frame, y_true, repeats=5, seed=42):
    """Mean drop in PR-AUC when each input is shuffled (higher = more important)."""
    rng = np.random.default_rng(seed)
    features = frame[C.FEATURE_COLUMNS].reset_index(drop=True)
    baseline = average_precision_score(y_true, predict_proba(model, features))
    groups = {C.SENSOR_LABEL[c]: [c] for c in C.SENSOR_COLUMNS}
    groups["Machine type"] = list(C.TYPE_COLUMNS)
    rows = []
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = features.copy()
            order = rng.permutation(len(features))
            shuffled[columns] = features[columns].to_numpy()[order]
            drops.append(baseline - average_precision_score(y_true, predict_proba(model, shuffled)))
        rows.append({"feature": name, "drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return baseline, pd.DataFrame(rows).sort_values("drop", ascending=False).reset_index(drop=True)
