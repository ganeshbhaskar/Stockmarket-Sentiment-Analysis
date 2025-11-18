import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEWS_FILE   = ROOT / "data" / "processed" / "news_AAPL.O_finbert.csv"
PRICES_FILE = ROOT / "data" / "raw"       / "prices_AAPL.O.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "panel_AAPL_O_sentiment_prices.csv"


def load_news():
    df = pd.read_csv(NEWS_FILE)

    # Ensure DATE exists
    if "DATE" not in df.columns:
        raise KeyError("FinBERT file must contain DATE column.")

    # Convert DATE → date only
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date

    # Keep required columns
    sentiment_cols = ["DATE", "FINBERT_SCORE", "FINBERT_LABEL",
                      "FINBERT_POS", "FINBERT_NEG", "FINBERT_NEU"]

    news = df[sentiment_cols].copy()

    # Aggregate to 1 row per date:
    daily = (
        news.groupby("DATE")
            .agg(
                DAILY_SENTIMENT=("FINBERT_SCORE", "mean"),
                NUM_HEADLINES=("FINBERT_SCORE", "count"),
                POS_SHARE=("FINBERT_LABEL", lambda x: (x == "positive").mean()),
                NEG_SHARE=("FINBERT_LABEL", lambda x: (x == "negative").mean()),
            )
            .reset_index()
    )

    return daily


def load_prices():
    df = pd.read_csv(PRICES_FILE)

    # Convert TIMESTAMP → DATE
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df["DATE"] = df["TIMESTAMP"].dt.date

    # Rename important fields
    rename = {
        "TRDPRC_1": "CLOSE",
        "HIGH_1":   "HIGH",
        "LOW_1":    "LOW",
        "OPEN_PRC": "OPEN",
        "ACVOL_UNS": "VOLUME"
    }
    df.rename(columns=rename, inplace=True)

    # Compute daily return
    df = df.sort_values("DATE")
    df["RETURN"] = df["CLOSE"].pct_change()

    return df


def main():
    print("\n📥 Loading sentiment...")
    news = load_news()
    print(f"News rows: {len(news)}")

    print("\n📈 Loading prices...")
    prices = load_prices()
    print(f"Price rows: {len(prices)}")

    # --- IMPORTANT: MERGE ONLY MATCHING DATES ---
    panel = prices.merge(news, on="DATE", how="inner")

    # Save result
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ PANEL MERGED SUCCESSFULLY")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Final rows: {len(panel)}")
    print("\nColumns:", list(panel.columns))
    print("\nPreview:\n", panel.head())


if __name__ == "__main__":
    main()
