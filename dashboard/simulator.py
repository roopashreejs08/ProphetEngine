"""A small engine simulator for the Live monitor.

It produces plausible AI4I-style readings: every parameter (medians, spread,
torque-speed relationship, air/process temperature gap, wear rate) is
estimated from the training split, so the stream stays inside the data the
models were trained on. Faults push readings toward the conditions that the
dataset documents as failure causes. Readings are synthetic; the model
scoring them is real.
"""

import numpy as np

from dashboard import config as C

FAULTS = {
    "overstrain": "Overstrain (high torque + worn tool)",
    "heat": "Heat dissipation (low speed, small temperature gap)",
    "power": "Power failure (very high speed, almost no torque)",
}


def training_stats(train):
    """Parameters for the simulator, all estimated from the training split."""
    sensors = {}
    for column in C.SENSOR_COLUMNS:
        series = train[column].astype(float)
        sensors[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)) or 1.0,
        }
    rpm = train["Rotational speed [rpm]"].to_numpy(dtype=float)
    torque = train["Torque [Nm]"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(rpm, torque, 1)
    residual_std = float(np.std(torque - (intercept + slope * rpm)))
    gap = (train["Process temperature [K]"] - train["Air temperature [K]"]).to_numpy(dtype=float)
    failure_wear = train.loc[train[C.TARGET] == 1, "Tool wear [min]"].to_numpy(dtype=float)
    wear_life = float(np.median(failure_wear)) if len(failure_wear) else sensors["Tool wear [min]"]["max"]
    return {
        "sensors": sensors,
        "torque_slope": float(slope),
        "torque_intercept": float(intercept),
        "torque_residual_std": residual_std,
        "gap_mean": float(gap.mean()),
        "gap_std": float(gap.std()),
        "wear_life": wear_life,
    }


def new_state(stats, machine_type="M", start_wear=0.0, seed=42):
    s = stats["sensors"]
    air = s["Air temperature [K]"]["median"]
    rpm = s["Rotational speed [rpm]"]["median"]
    return {
        "machine_type": machine_type,
        "air": air,
        "rpm": rpm,
        "wear": float(np.clip(start_wear, s["Tool wear [min]"]["min"], s["Tool wear [min]"]["max"])),
        "fault": None,
        "fault_level": 0.0,  # ramps 0 -> 1 so faults develop gradually
        "rng": np.random.default_rng(seed),
        "tick": 0,
    }


def _clip(stats, column, value):
    s = stats["sensors"][column]
    return float(np.clip(value, s["min"], s["max"]))


def step(state, stats, wear_per_tick=2.0):
    """Advance one tick and return a reading dict keyed by feature column."""
    s = stats["sensors"]
    rng = state["rng"]
    air_med = s["Air temperature [K]"]["median"]
    rpm_med = s["Rotational speed [rpm]"]["median"]

    state["air"] = _clip(stats, "Air temperature [K]",
                         state["air"] + rng.normal(0, s["Air temperature [K]"]["std"] * 0.05)
                         + 0.03 * (air_med - state["air"]))
    state["rpm"] = _clip(stats, "Rotational speed [rpm]",
                         state["rpm"] + rng.normal(0, s["Rotational speed [rpm]"]["std"] * 0.08)
                         + 0.10 * (rpm_med - state["rpm"]))
    state["wear"] = _clip(stats, "Tool wear [min]", state["wear"] + wear_per_tick)
    if state["fault"]:
        state["fault_level"] = min(1.0, state["fault_level"] + 0.12)
    else:
        state["fault_level"] = max(0.0, state["fault_level"] - 0.25)
    level = state["fault_level"]
    fault = state["fault"]

    rpm = state["rpm"]
    air = state["air"]
    gap = stats["gap_mean"] + rng.normal(0, stats["gap_std"] * 0.3)
    wear = state["wear"]

    # Faults blend the reading toward an operating point the dataset documents as a
    # failure cause; fault_level ramps up so the change develops over a few ticks.
    target_torque = None
    if fault == "heat":  # hot air, small air/process gap, low speed
        air = air + level * (302.8 - air)
        gap = gap + level * (8.0 - gap)
        rpm = rpm + level * (1320 - rpm)
        target_torque = 50.0
    elif fault == "power":  # power = torque x speed drops below 3,500 W
        rpm = rpm + level * (2800 - rpm)
        target_torque = 9.0
    elif fault == "overstrain":  # worn tool under heavy torque
        rpm = rpm + level * (1300 - rpm)
        target_torque = 62.0
        wear = wear + level * (220 - wear) if wear < 220 else wear

    torque = stats["torque_intercept"] + stats["torque_slope"] * rpm + rng.normal(0, stats["torque_residual_std"] * 0.3)
    if target_torque is not None:
        torque = torque + level * (target_torque - torque)

    state["tick"] += 1
    return {
        "Air temperature [K]": _clip(stats, "Air temperature [K]", air),
        "Process temperature [K]": _clip(stats, "Process temperature [K]", air + gap),
        "Rotational speed [rpm]": _clip(stats, "Rotational speed [rpm]", rpm),
        "Torque [Nm]": _clip(stats, "Torque [Nm]", torque),
        "Tool wear [min]": _clip(stats, "Tool wear [min]", wear),
    }
