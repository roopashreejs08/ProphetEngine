"""Plain data helpers (no Streamlit imports, so they are easy to unit test).

Loaders return ``None`` when an artifact is missing or malformed; the UI turns
that into a friendly message instead of a stack trace.
"""

import io
import json

import numpy as np
import pandas as pd

from dashboard import config as C


# ---------------------------------------------------------------------------
# Repository artifacts
# ---------------------------------------------------------------------------
def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_model_frame(path):
    """Read an AI4I split (features + target) and check its schema."""
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    required = C.FEATURE_COLUMNS + [C.TARGET]
    if frame.empty or not set(required).issubset(frame.columns):
        return None
    if not all(pd.api.types.is_numeric_dtype(frame[col]) for col in required):
        return None
    return frame


def read_metrics(path):
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    if "Model" not in frame.columns:
        return None
    return frame


def read_importance(path):
    """The importance CSV is a saved pandas Series: unnamed index column + values."""
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    if frame.shape[1] < 2:
        return None
    frame = frame.iloc[:, :2]
    frame.columns = ["Feature", "Importance"]
    frame["Importance"] = pd.to_numeric(frame["Importance"], errors="coerce")
    frame = frame.dropna().sort_values("Importance", ascending=False)
    return frame.reset_index(drop=True)


def machine_type_from_onehot(frame):
    result = pd.Series("M", index=frame.index, dtype=object)
    result[frame["Type_L"] == 1] = "L"
    result[frame["Type_H"] == 1] = "H"
    return result


def attach_failure_modes(split, cleaned):
    """Recover failure-mode flags for each row of a split.

    The model splits drop TWF/HDF/PWF/OSF/RNF to avoid leakage, but every row
    is unique on its features, so we can join back to ``ai4i_cleaned.csv`` for
    *descriptive* analysis (which kinds of failure the model catches).
    """
    if split is None or cleaned is None:
        return None
    modes = list(C.FAILURE_MODES)
    keys = C.FEATURE_COLUMNS + [C.TARGET]
    if not set(keys + modes).issubset(cleaned.columns):
        return None
    lookup = cleaned[keys + modes].drop_duplicates(subset=keys)
    merged = split.merge(lookup, on=keys, how="left", validate="many_to_one")
    if len(merged) != len(split):
        return None
    merged[modes] = merged[modes].fillna(0).astype(int)
    return merged


def failure_mode_label(row):
    if row.get(C.TARGET, 0) != 1:
        return "None"
    names = [code for code in C.FAILURE_MODES if row.get(code, 0) == 1]
    if names:
        return ", ".join(names)
    return "Unlabelled failure"


def training_ranges(train):
    ranges = {}
    for column in C.SENSOR_COLUMNS:
        series = train[column].astype(float)
        ranges[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "p05": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)) or 1.0,
        }
    return ranges


def build_feature_row(values, machine_type):
    """One model-ready row from sensor values and an L/M/H machine type."""
    row = {column: float(values[column]) for column in C.SENSOR_COLUMNS}
    for kind in C.MACHINE_TYPES:
        row[f"Type_{kind}"] = int(machine_type == kind)
    return pd.DataFrame([row], columns=C.FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Uploaded CSV validation (batch scoring)
# ---------------------------------------------------------------------------
MAX_UPLOAD_ROWS = 50_000

# Accepted header spellings -> canonical feature column.
_ALIASES = {
    "air temperature [k]": "Air temperature [K]",
    "air_temperature": "Air temperature [K]",
    "air temperature": "Air temperature [K]",
    "process temperature [k]": "Process temperature [K]",
    "process_temperature": "Process temperature [K]",
    "process temperature": "Process temperature [K]",
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "rotational_speed": "Rotational speed [rpm]",
    "rotational speed": "Rotational speed [rpm]",
    "torque [nm]": "Torque [Nm]",
    "torque": "Torque [Nm]",
    "tool wear [min]": "Tool wear [min]",
    "tool_wear": "Tool wear [min]",
    "tool wear": "Tool wear [min]",
    "type": "machine_type",
    "machine_type": "machine_type",
    "machine type": "machine_type",
    "machine failure": C.TARGET,
    "machine_failure": C.TARGET,
    "type_h": "Type_H",
    "type_l": "Type_L",
    "type_m": "Type_M",
}


class UploadResult:
    """Outcome of validating an uploaded CSV."""

    def __init__(self, features=None, labels=None, original=None, errors=None, warnings=None):
        self.features = features
        self.labels = labels
        self.original = original
        self.errors = errors or []
        self.warnings = warnings or []

    @property
    def ok(self):
        return not self.errors and self.features is not None


def template_csv():
    sample = pd.DataFrame(
        [
            [298.1, 308.6, 1551, 42.8, 0, "M"],
            [302.4, 310.9, 1316, 66.1, 212, "L"],
            [300.8, 309.4, 2678, 10.7, 86, "H"],
        ],
        columns=C.SENSOR_COLUMNS + ["Type"],
    )
    return sample.to_csv(index=False).encode("utf-8")


def validate_upload(source, ranges=None):
    """Validate and normalise an uploaded CSV into model features.

    ``source`` is a path, bytes or a file-like object. Accepts the processed
    split format (Type_H/Type_L/Type_M), the raw AI4I format (a ``Type``
    column with L/M/H) and the live-feed format (snake_case headers).
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        raw = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        return UploadResult(errors=["The file is empty."])
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return UploadResult(errors=["The file could not be read as CSV. Save it as comma-separated UTF-8 and try again."])

    if raw.empty:
        return UploadResult(errors=["The file has a header but no rows."])
    if len(raw) > MAX_UPLOAD_ROWS:
        return UploadResult(errors=[f"The file has {len(raw):,} rows; the limit is {MAX_UPLOAD_ROWS:,}."])

    renamed = {}
    for column in raw.columns:
        key = str(column).strip().lower()
        if key in _ALIASES:
            renamed[column] = _ALIASES[key]
    frame = raw.rename(columns=renamed)
    if frame.columns.duplicated().any():
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        return UploadResult(errors=["The same column appears more than once: " + ", ".join(dupes)])

    missing = [column for column in C.SENSOR_COLUMNS if column not in frame.columns]
    has_onehot = set(C.TYPE_COLUMNS).issubset(frame.columns)
    has_type = "machine_type" in frame.columns
    if not (has_onehot or has_type):
        missing.append("Type (L/M/H) or Type_H, Type_L, Type_M")
    if missing:
        return UploadResult(errors=["Missing required columns: " + "; ".join(missing)])

    errors, warnings = [], []
    numeric = frame[C.SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_numeric = ~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)

    if has_type:
        types = frame["machine_type"].astype(str).str.strip().str.upper()
        bad_type = ~types.isin(C.MACHINE_TYPES)
        onehot = pd.DataFrame({f"Type_{t}": (types == t).astype(int) for t in ["H", "L", "M"]})
    else:
        onehot = frame[C.TYPE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        bad_type = ~(onehot.isin([0, 1]).all(axis=1) & (onehot.sum(axis=1) == 1))
        onehot = onehot.fillna(0).astype(int)

    bad = bad_numeric | bad_type.to_numpy()
    if bad.all():
        return UploadResult(errors=["No row has valid numeric sensor values and a machine type of L, M or H."])
    if bad.any():
        rows = ", ".join(str(i + 2) for i in np.flatnonzero(bad)[:10])
        more = " ..." if bad.sum() > 10 else ""
        warnings.append(f"Skipped {int(bad.sum())} invalid row(s) (CSV line numbers {rows}{more}).")

    keep = ~bad
    features = pd.concat([numeric[keep].reset_index(drop=True), onehot[keep].reset_index(drop=True)], axis=1)
    features = features[C.FEATURE_COLUMNS].astype(float)

    if ranges:
        outside = []
        for column in C.SENSOR_COLUMNS:
            lo, hi = ranges[column]["min"], ranges[column]["max"]
            count = int(((features[column] < lo) | (features[column] > hi)).sum())
            if count:
                outside.append(f"{C.SENSOR_LABEL[column]} ({count})")
        if outside:
            warnings.append(
                "Some values fall outside the range seen in training, so predictions there are extrapolations: "
                + ", ".join(outside) + "."
            )

    labels = None
    if C.TARGET in frame.columns:
        target = pd.to_numeric(frame.loc[keep, C.TARGET], errors="coerce").reset_index(drop=True)
        if target.isin([0, 1]).all():
            labels = target.astype(int)
        else:
            warnings.append("The 'Machine failure' column is not all 0/1, so accuracy metrics are not shown.")

    original = raw[keep].reset_index(drop=True)
    return UploadResult(features=features, labels=labels, original=original, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Live feed (data/live/stream.csv)
# ---------------------------------------------------------------------------
def parse_stream_csv(path):
    """Parse the optional external telemetry feed. Returns a status dict."""
    result = {"status": "ok", "frame": None, "warnings": [], "newest": None}
    if not path.exists():
        result["status"] = "missing"
        return result
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        result["status"] = "empty"
        return result
    except (OSError, ValueError):
        result["status"] = "unreadable"
        return result

    raw.columns = [str(column).strip() for column in raw.columns]
    missing = [c for c in C.STREAM_COLUMNS if c not in raw.columns]
    extra = [c for c in raw.columns if c not in C.STREAM_COLUMNS]
    if missing or extra or raw.columns.duplicated().any():
        result["status"] = "invalid_columns"
        if missing:
            result["warnings"].append("Missing columns: " + ", ".join(missing))
        if extra:
            result["warnings"].append("Unexpected columns: " + ", ".join(extra))
        return result
    if raw.empty:
        result["status"] = "empty"
        return result

    timestamps = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    numeric = {c: pd.to_numeric(raw[c], errors="coerce") for c in C.STREAM_SENSOR_MAP}
    machine_type = raw["machine_type"].astype(str).str.strip().str.upper()
    valid = timestamps.notna() & machine_type.isin(C.MACHINE_TYPES)
    for column in C.STREAM_SENSOR_MAP:
        valid &= np.isfinite(numeric[column].to_numpy(dtype=float))

    invalid = int((~valid).sum())
    if invalid:
        rows = ", ".join(str(i + 2) for i in raw.index[~valid][:8])
        result["warnings"].append(f"Skipped {invalid} malformed row(s). CSV line numbers: {rows}" + (" ..." if invalid > 8 else ""))
    if not valid.any():
        result["status"] = "no_valid_rows"
        return result

    frame = pd.DataFrame({"timestamp": timestamps[valid], "machine_type": machine_type[valid]})
    for column, target in C.STREAM_SENSOR_MAP.items():
        frame[target] = numeric[column][valid].astype(float)
    frame = frame.reset_index(drop=True)
    result["frame"] = frame
    result["newest"] = frame["timestamp"].max()
    return result
