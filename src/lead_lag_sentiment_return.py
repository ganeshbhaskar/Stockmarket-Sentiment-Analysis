import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE        = ROOT / "data" / "processed" / "panel_AAPL_O_sentiment_prices.csv"
LEAD_LAG_CSV      = ROOT / "data" / "processed" / "lead_lag_sentiment_return.csv"
LEAD_LAG_PLOT_PNG = ROOT / "data" / "processed" / "lead_lag_sentiment_return_linechart.png"

# date range you’ve been using
START_DATE = "2025-08-08"
END_DATE   = "2025-11-17"
# -----------------------------


def main():
    print(f"📄 Loading panel → {PANEL_FILE}")
    df = pd.read_csv(PANEL_FILE)

    # Ensure DATE + numeric types
    if "DATE" not in df.columns:
        raise KeyError("Expected a DATE column in the panel file.")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df["RETURN"] = pd.to_numeric(df["RETURN"], errors="coerce")
    df["DAILY_SENTIMENT"] = pd.to_numeric(df["DAILY_SENTIMENT"], errors="coerce")

    # Filter to the chosen date range
    mask = (df["DATE"] >= START_DATE) & (df["DATE"] <= END_DATE)
    df = df.loc[mask].sort_values("DATE").reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows after date filtering – check START_DATE / END_DATE.")

    # -------- 1) BUILD LEAD/LAG SERIES --------
    # Sentiment at day t
    # Return at day t+1 (shifted up / negative index shift)
    df["RETURN_TOMORROW"] = df["RETURN"].shift(-1)

    # Drop last row (no tomorrow return)
    lead_lag = df[["DATE", "DAILY_SENTIMENT", "RETURN_TOMORROW"]].dropna().copy()

    if lead_lag.empty:
        raise ValueError("Lead/lag dataframe is empty – check RETURN column.")

    # -------- 2) GLOBAL CORRELATION --------
    corr_val = lead_lag["DAILY_SENTIMENT"].corr(lead_lag["RETURN_TOMORROW"])
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("📈 LEAD/LAG CORRELATION")
    print("Sentiment today  vs  Return tomorrow")
    print(f"Correlation(DAILY_SENTIMENT_t, RETURN_t+1) = {corr_val:.4f}")
    print("━━━━━━━━━━━━━━━━━━━━━━\n")

    # -------- 3) SAVE CSV --------
    LEAD_LAG_CSV.parent.mkdir(parents=True, exist_ok=True)
    lead_lag.to_csv(LEAD_LAG_CSV, index=False)
    print(f"💾 Lead/lag data saved → {LEAD_LAG_CSV}")

    # -------- 4) LINE CHART (like earlier) --------
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Left axis: RETURN_TOMORROW
    ax1.plot(
        lead_lag["DATE"],
        lead_lag["RETURN_TOMORROW"],
        color="green",
        linewidth=2,
        label="Return (Tomorrow)",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Return Tomorrow", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    # Right axis: DAILY_SENTIMENT
    ax2 = ax1.twinx()
    ax2.plot(
        lead_lag["DATE"],
        lead_lag["DAILY_SENTIMENT"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Daily Sentiment (Today)",
    )
    ax2.set_ylabel("Daily Sentiment", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Title + layout
    plt.title("Lead/Lag: Sentiment Today vs Return Tomorrow (AAPL.O)")
    fig.tight_layout()

    LEAD_LAG_PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(LEAD_LAG_PLOT_PNG, dpi=300)
    plt.close()

    print(f"📉 Lead/lag line chart saved → {LEAD_LAG_PLOT_PNG}")


if __name__ == "__main__":
    main()
