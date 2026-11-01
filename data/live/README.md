# Live telemetry feed

`data/live/stream.csv` is an optional external telemetry file watched by the
Live prediction page. Rows are scored in file order as they are appended.

## Format

The CSV must contain exactly these columns:

```text
timestamp,air_temperature,process_temperature,rotational_speed,torque,tool_wear,machine_type
```

- `timestamp`: ISO 8601 timestamp in UTC, for example `2026-10-30T14:03:05Z`.
- `air_temperature`: air temperature in K.
- `process_temperature`: process temperature in K.
- `rotational_speed`: rotational speed in rpm.
- `torque`: torque in Nm.
- `tool_wear`: tool wear in minutes.
- `machine_type`: `L`, `M`, or `H`.

All sensor values must be finite numbers. Rows with an invalid timestamp, a
missing or non-numeric sensor value, or an unsupported machine type are skipped
with a visible warning. The feed is treated as stale when its newest valid row
is more than 30 seconds old. The app never fills in missing values.

If the file is absent, create it with the header above:

```bash
mkdir -p data/live
printf 'timestamp,air_temperature,process_temperature,rotational_speed,torque,tool_wear,machine_type\n' > data/live/stream.csv
```

Append one complete row per reading. The page scores new rows automatically.
