
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "report" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_sensor_trends(dataset="FD001", engine_ids=(1, 2, 3)):
    path = DATA_DIR / f"{dataset}_train.csv"
    df = pd.read_csv(path)

    sensors = [
        c for c in df.columns
        if c.startswith("sensor_")
    ]

    if not sensors:
        raise ValueError("No sensor columns found.")

    # Plot several sensors to make degradation patterns visible.
    selected_sensors = sensors[:4]

    fig, axes = plt.subplots(
        len(selected_sensors), 1,
        figsize=(10, 3 * len(selected_sensors)),
        sharex=True
    )

    if len(selected_sensors) == 1:
        axes = [axes]

    for ax, sensor in zip(axes, selected_sensors):
        for engine_id in engine_ids:
            engine = df[df["engine_id"] == engine_id].sort_values("cycle")
            if not engine.empty:
                ax.plot(
                    engine["cycle"], engine[sensor],
                    label=f"Engine {engine_id}"
                )

        ax.set_ylabel(sensor)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Engine cycle")
    fig.suptitle(f"{dataset}: Sensor trends by engine")
    fig.tight_layout()

    output = FIGURE_DIR / f"{dataset}_sensor_trends.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output)


if __name__ == "__main__":
    plot_sensor_trends("FD001")
    plot_sensor_trends("FD003")
