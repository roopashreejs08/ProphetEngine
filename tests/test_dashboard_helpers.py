"""Unit tests for the dashboard's data and scoring helpers."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from dashboard import config as C
from dashboard import data as D
from dashboard import scoring as S
from dashboard import simulator as SIM


@pytest.fixture(scope="module")
def validation():
    frame = D.read_model_frame(C.VALIDATION_PATH)
    assert frame is not None
    return frame


@pytest.fixture(scope="module")
def train():
    return D.read_model_frame(C.TRAIN_PATH)


@pytest.fixture(scope="module")
def forest():
    return joblib.load(C.RESULTS_DIR / "random_forest.joblib")


# --- repository artifacts -------------------------------------------------
def test_saved_metrics_reproduce_from_artifacts(validation):
    """The dashboard's live numbers must match results/classical_ml_metrics.csv."""
    saved = D.read_metrics(C.METRICS_PATH).set_index("Model")
    y = validation[C.TARGET].to_numpy()
    for label in C.BASE_MODELS:
        model = joblib.load(C.RESULTS_DIR / C.MODEL_FILES[label])
        m = S.metrics_at_threshold(y, S.predict_proba(model, validation), 0.5)
        assert m["pr_auc"] == pytest.approx(saved.loc[label, "Average Precision (PR-AUC)"], abs=1e-9)
        assert m["roc_auc"] == pytest.approx(saved.loc[label, "ROC-AUC"], abs=1e-9)


def test_tuned_threshold_matches_readme(validation, forest):
    y = validation[C.TARGET].to_numpy()
    m = S.metrics_at_threshold(y, S.predict_proba(forest, validation), C.TUNED_THRESHOLD)
    assert (m["tp"], m["failures"]) == (50, 68)
    assert m["accuracy"] == pytest.approx(0.9745)


def test_importance_loader_reads_series_csv():
    frame = D.read_importance(C.IMPORTANCE_PATH)
    assert list(frame.columns) == ["Feature", "Importance"]
    assert frame.iloc[0]["Feature"] == "Rotational speed [rpm]"
    assert frame["Importance"].sum() == pytest.approx(1.0, abs=1e-6)


def test_failure_modes_join_back(validation):
    cleaned = pd.read_csv(C.CLEANED_PATH)
    merged = D.attach_failure_modes(validation, cleaned)
    assert len(merged) == len(validation)
    assert merged[C.TARGET].sum() == 68
    failures = merged[merged[C.TARGET] == 1]
    assert failures["HDF"].sum() == 29 and failures["OSF"].sum() == 16
    assert D.failure_mode_label(merged.iloc[0]) == "None"


def test_missing_files_return_none(tmp_path):
    assert D.read_model_frame(tmp_path / "nope.csv") is None
    assert D.read_json(tmp_path / "nope.json") is None
    assert D.read_importance(tmp_path / "nope.csv") is None


# --- scoring --------------------------------------------------------------
def test_metrics_at_threshold_counts():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.7, 0.9])
    m = S.metrics_at_threshold(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    # Threshold is inclusive.
    assert S.metrics_at_threshold(y, p, 0.4)["tp"] == 3


def test_metrics_rejects_mismatched_input():
    with pytest.raises(ValueError):
        S.metrics_at_threshold([0, 1], [0.5], 0.5)


def test_sweep_and_cost_pick_cheapest_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.55, 0.5, 0.95])
    sweep = S.add_costs(S.threshold_sweep(y, p, thresholds=[0.1, 0.3, 0.5, 0.6]), cost_missed=100, cost_false_alarm=1)
    best = S.best_threshold(sweep)
    # 0.3 and 0.5 both catch every failure with a single false alarm; the first wins.
    assert best["threshold"] == 0.3
    assert best["cost"] == 1
    assert sweep.set_index("threshold").loc[0.6, "cost"] == 100


def test_risk_tiers():
    tiers = S.risk_tiers([0.05, 0.15, 0.3], 0.23)
    assert list(tiers) == ["healthy", "watch", "critical"]
    assert S.risk_tier(None, 0.5) == "unknown"


def test_physics_checks_flag_overstrain_and_power():
    values = {"Air temperature [K]": 300, "Process temperature [K]": 310, "Rotational speed [rpm]": 1300,
              "Torque [Nm]": 65, "Tool wear [min]": 210}
    checks = {c["code"]: c["triggered"] for c in S.physics_checks(values, "L")}
    assert checks["OSF"]  # 210 * 65 = 13,650 > 11,000
    assert not checks["PWF"]  # 65 * 1300 * 2pi/60 = 8,849 W, inside 3,500-9,000
    assert checks["TWF"]
    assert not checks["HDF"]  # gap is 10 K


def test_local_contributions_and_sensitivity(train, forest):
    ranges = D.training_ranges(train)
    risky = {"Air temperature [K]": 300, "Process temperature [K]": 310, "Rotational speed [rpm]": 1300,
             "Torque [Nm]": 65, "Tool wear [min]": 215}
    prob, contrib = S.local_contributions(forest, risky, "L", ranges)
    assert 0 <= prob <= 1
    assert set(contrib["column"]) == set(C.SENSOR_COLUMNS)
    grid = np.linspace(0, 250, 11)
    curve = S.sensitivity(forest, risky, "L", "Tool wear [min]", grid)
    assert curve.shape == (11,) and ((curve >= 0) & (curve <= 1)).all()


# --- uploads --------------------------------------------------------------
def test_upload_accepts_raw_ai4i_format(train):
    csv = D.template_csv()
    result = D.validate_upload(csv, D.training_ranges(train))
    assert result.ok
    assert list(result.features.columns) == C.FEATURE_COLUMNS
    assert result.features[C.TYPE_COLUMNS].sum(axis=1).eq(1).all()


def test_upload_accepts_processed_split():
    result = D.validate_upload(C.VALIDATION_PATH.read_bytes())
    assert result.ok and len(result.features) == 2000
    assert result.labels is not None and result.labels.sum() == 68


def test_upload_accepts_feed_format():
    csv = ("timestamp,air_temperature,process_temperature,rotational_speed,torque,tool_wear,machine_type\n"
           "2026-10-30T14:03:05Z,300.1,310.2,1500,40,100,m\n")
    result = D.validate_upload(csv.encode())
    assert result.ok and result.features.loc[0, "Type_M"] == 1


def test_upload_reports_missing_columns():
    result = D.validate_upload(b"Torque [Nm],Type\n40,L\n")
    assert not result.ok
    assert "Missing required columns" in result.errors[0]


def test_upload_skips_bad_rows_and_warns_out_of_range(train):
    csv = io.StringIO()
    pd.DataFrame({
        "Air temperature [K]": [300, "hot", 300],
        "Process temperature [K]": [310, 310, 310],
        "Rotational speed [rpm]": [1500, 1500, 9999],
        "Torque [Nm]": [40, 40, 40],
        "Tool wear [min]": [10, 10, 10],
        "Type": ["L", "M", "X"],
    }).to_csv(csv, index=False)
    result = D.validate_upload(csv.getvalue().encode(), D.training_ranges(train))
    assert result.ok and len(result.features) == 1
    assert any("Skipped 2" in w for w in result.warnings)


def test_upload_empty_and_garbage():
    assert not D.validate_upload(b"").ok
    assert not D.validate_upload(b"a,b\n").ok


# --- live feed & simulator ------------------------------------------------
def test_parse_stream_csv(tmp_path):
    path = tmp_path / "stream.csv"
    assert D.parse_stream_csv(path)["status"] == "missing"
    path.write_text(",".join(C.STREAM_COLUMNS) + "\n"
                    "2026-10-30T14:03:05Z,300,310,1500,40,100,L\n"
                    "not-a-time,300,310,1500,40,100,L\n")
    parsed = D.parse_stream_csv(path)
    assert parsed["status"] == "ok" and len(parsed["frame"]) == 1
    assert parsed["warnings"]


def test_simulator_stays_in_training_range(train):
    stats = SIM.training_stats(train)
    state = SIM.new_state(stats, machine_type="L")
    state["fault"] = "overstrain"
    for _ in range(50):
        reading = SIM.step(state, stats)
    for column in C.SENSOR_COLUMNS:
        s = stats["sensors"][column]
        assert s["min"] <= reading[column] <= s["max"]
