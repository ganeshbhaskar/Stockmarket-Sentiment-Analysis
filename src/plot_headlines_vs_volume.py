import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   # (still fine, though not used here)
import matplotlib.dates as mdates
from pathlib import Path

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE   = ROOT / "data" / "processed" / "panel_AAPL_O_sentiment_prices.csv"
SCATTER_PLOT = ROOT / "data" / "processed" / "headlines_vs_volume_scatter.png"
LINE_PLOT    = ROOT / "data" / "processed" / "headlines_vs_volume_line.png"

START_DATE = pd.Timestamp("2025-08-08")
END_DATE   = pd.Timestamp("2025-11-17")
# ----------------------------------------


def main():
    print(f"📄 Loading panel → {PANEL_FILE}")
    df = pd.read_csv(PANEL_FILE)

    # Prepare dataset
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")
    df["NUM_HEADLINES"] = pd.to_numeric(df["NUM_HEADLINES"], errors="coerce")

    # Filter the period
    df = df[(df["DATE"] >= START_DATE) & (df["DATE"] <= END_DATE)]

    # ---------- DUAL-AXIS LINE CHART ----------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # NUM_HEADLINES line (left axis)
    line1, = ax1.plot(
        df["DATE"],
        df["NUM_HEADLINES"],
        color="purple",
        linewidth=2,
        marker="o",
        label="NUM_HEADLINES",
    )
    ax1.set_ylabel("Number of Headlines", color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")

    # VOLUME line on secondary Y-axis (right axis)
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        df["DATE"],
        df["VOLUME"],
        color="green",
        linewidth=2,
        label="VOLUME",
    )
    ax2.set_ylabel("Trading Volume", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # ---------- Weekly ticks ----------
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=45)

    # ---------- ADD LEGEND (IMPORTANT) ----------
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Daily Headlines vs Trading Volume (AAPL.O)")
    fig.tight_layout()

    LINE_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(LINE_PLOT, dpi=300)
    plt.close()

    print(f"📌 Line chart saved → {LINE_PLOT}")


if __name__ == "__main__":
    main()
