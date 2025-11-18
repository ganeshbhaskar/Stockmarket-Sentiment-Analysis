import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE      = ROOT / "data" / "processed" / "panel_AAPL_O_sentiment_prices.csv"
CORR_TABLE      = ROOT / "data" / "processed" / "correlation_table.csv"
CORR_PLOT       = ROOT / "data" / "processed" / "correlation_heatmap.png"
DAILY_CORR_FILE = ROOT / "data" / "processed" / "daily_correlation.csv"
DAILY_CORR_PLOT = ROOT / "data" / "processed" / "rolling_corr_return_daily_sentiment.png"

# Date range for analysis (matches your panel window)
START_DATE = pd.Timestamp("2025-08-08")
END_DATE   = pd.Timestamp("2025-11-17")

# Rolling window size in calendar days / rows
ROLL_WINDOW_DAYS = 5   # you can change to 10, 20, etc.


def main():
    print(f"📄 Loading panel → {PANEL_FILE}")
    df = pd.read_csv(PANEL_FILE)

    # Ensure DATE exists and is datetime
    if "DATE" not in df.columns:
        raise KeyError("Expected a DATE column in the panel file.")
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Filter to the desired date range
    mask = (df["DATE"] >= START_DATE) & (df["DATE"] <= END_DATE)
    df = df.loc[mask].sort_values("DATE").reset_index(drop=True)
    print(f"📆 Filtered rows: {len(df)} (from {START_DATE.date()} to {END_DATE.date()})")

    # We'll use a clean subset for correlation work
    # Keep the main price metrics + sentiment metrics
    price_cols = [
        "CLOSE", "HIGH", "LOW", "OPEN", "VOLUME",
        "BID", "ASK", "TRNOVR_UNS", "VWAP",
        "BLKCOUNT", "BLKVOLUM", "NUM_MOVES",
        "NAVALUE", "VWAP_VOL", "RETURN",
    ]
    sentiment_cols = ["DAILY_SENTIMENT", "POS_SHARE", "NEG_SHARE", "NUM_HEADLINES"]

    # Only keep columns that actually exist in the panel
    price_cols     = [c for c in price_cols if c in df.columns]
    sentiment_cols = [c for c in sentiment_cols if c in df.columns]

    work_cols = ["DATE"] + price_cols + sentiment_cols
    df = df[work_cols].copy()

    # Drop rows where we don't have a RETURN value
    # (first row will typically have RETURN NaN)
    df = df[df["RETURN"].notna()].reset_index(drop=True)

    # -------- 1) GLOBAL CORRELATION (over this period) --------
    num_cols = [c for c in df.columns if c != "DATE"]
    print("🔎 Numeric fields considered for GLOBAL correlation:")
    print(num_cols)

    corr = df[num_cols].corr()

    CORR_TABLE.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(CORR_TABLE)
    print(f"📁 Correlation table saved → {CORR_TABLE}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap: Sentiment vs Price Metrics")
    plt.tight_layout()
    plt.savefig(CORR_PLOT, dpi=300)
    plt.close()
    print(f"📊 Correlation heatmap saved → {CORR_PLOT}")

    if "RETURN" in df.columns and "DAILY_SENTIMENT" in df.columns:
        r = df["RETURN"].corr(df["DAILY_SENTIMENT"])
        print("\n━━━━━━━━━━━━━━━━━━━━━━")
        print("📈 GLOBAL KEY RELATIONSHIP (over full period)")
        print(f"Correlation(DAILY_SENTIMENT, RETURN) = {r:.4f}")
        print("━━━━━━━━━━━━━━━━━━━━━━")

    # -------- 2) ROLLING DAILY CORRELATIONS --------
    # For each day, correlate RETURN with each sentiment metric
    for col in sentiment_cols:
        new_name = f"RCORR_RETURN_{col}_{ROLL_WINDOW_DAYS}d"
        # Rolling correlation uses the last N calendar days in the window
        df[new_name] = df["RETURN"].rolling(ROLL_WINDOW_DAYS).corr(df[col])
        print(f"✅ Added rolling correlation column: {new_name}")

    # Save per-day rolling correlation table
    keep_cols = ["DATE"] + [c for c in df.columns if c.startswith("RCORR_RETURN_")]
    daily_corr = df[keep_cols].copy()
    daily_corr.to_csv(DAILY_CORR_FILE, index=False)
    print(f"📁 Daily rolling correlations saved → {DAILY_CORR_FILE}")

    # Optional: plot RETURN vs DAILY_SENTIMENT rolling corr
    key_col = f"RCORR_RETURN_DAILY_SENTIMENT_{ROLL_WINDOW_DAYS}d"
    if key_col in df.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(df["DATE"], df[key_col], marker="o")
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"{ROLL_WINDOW_DAYS}-Day Rolling Correlation: RETURN vs DAILY_SENTIMENT")
        plt.xlabel("Date")
        plt.ylabel("Correlation")
        plt.tight_layout()
        plt.savefig(DAILY_CORR_PLOT, dpi=300)
        plt.close()
        print(f"📉 Rolling correlation plot saved → {DAILY_CORR_PLOT}")
    else:
        print("ℹ️ No DAILY_SENTIMENT rolling correlation column found to plot.")


if __name__ == "__main__":
    main()
