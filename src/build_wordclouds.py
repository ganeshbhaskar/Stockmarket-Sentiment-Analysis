# build_wordclouds.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# ------------ CONFIG ------------
ROOT = Path(__file__).resolve().parents[1]

NEWS_FILE   = ROOT / "data" / "processed" / "news_AAPL.O_finbert.csv"
OUTPUT_DIR  = ROOT / "data" / "processed" / "wordclouds"

# Adjust these if you want a different window
START_DATE = pd.Timestamp("2025-08-22")
END_DATE   = pd.Timestamp("2025-11-15")

# Sentiment thresholds for "very positive/negative"
POS_THRESHOLD = 0.30
NEG_THRESHOLD = -0.30
# --------------------------------


def make_wordcloud(text: str, title: str, out_path: Path):
    """Generate and save a word cloud with a Top Words bar legend."""
    if not text or not text.strip():
        print(f"⚠️ Empty text for '{title}', skipping word cloud.")
        return

    # ---- Stopwords (LSEG + Apple specific noise) ----
    custom_stopwords = STOPWORDS | {
        "Apple", "AAPL", "Inc", "says", "said", "Reuters",
        "NS", "RTTRS", "RTS", "NEWS", "BUSST", "HINDU", "CNBC",
        "ZACKS", "Ltd", "Corp"
    }

    # ---- Build word cloud ----
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=custom_stopwords,
        collocations=True,
    ).generate(text)

    # ---- Extract top words (normalized frequency 0–1) ----
    word_freq = wc.words_  # dict: word -> normalized freq
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

    words = [w for w, _ in top_words]
    freqs = [f for _, f in top_words]

    # ---- Figure with 2 panels: word cloud + bar legend ----
    fig, (ax_wc, ax_leg) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Left: word cloud
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title(title, fontsize=20)

    # Right: horizontal bar chart of top words
    y_pos = range(len(words))
    ax_leg.barh(y_pos, freqs, color="purple", alpha=0.8)
    ax_leg.set_yticks(y_pos)
    ax_leg.set_yticklabels(words)
    ax_leg.invert_yaxis()  # highest at top
    ax_leg.set_xlabel("Normalized Frequency")
    ax_leg.set_title("Top Words")

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼 Word cloud saved → {out_path}")


def main():
    print(f"📄 Loading news → {NEWS_FILE}")
    df = pd.read_csv(NEWS_FILE)
    print("📌 Columns:", list(df.columns))

    # ---- Parse DATE (your file is dd-mm-yyyy) ----
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DATE"])

    print(
        f"📅 Date range in file: {df['DATE'].min().date()} → {df['DATE'].max().date()}"
    )

    # ---- Filter to window ----
    mask = (df["DATE"] >= START_DATE) & (df["DATE"] <= END_DATE)
    df = df.loc[mask].copy()
    print(
        f"🕒 Rows after date filter {START_DATE.date()} → {END_DATE.date()}: {len(df)}"
    )

    # ---- Keep rows with real headlines ----
    df["HEADLINE"] = df["HEADLINE"].astype(str)
    df = df[df["HEADLINE"].str.strip() != ""]
    print(f"📰 Rows with non-empty HEADLINE: {len(df)}")

    # ==================================================
    # 1) ALL HEADLINES
    # ==================================================
    all_text = " ".join(df["HEADLINE"].tolist())
    make_wordcloud(
        all_text,
        "All Headlines (Aug–Nov 2025)",
        OUTPUT_DIR / "wordcloud_all_with_legend.png",
    )

    # ==================================================
    # 2) VERY POSITIVE HEADLINES
    # ==================================================
    if "FINBERT_SCORE" in df.columns:
        pos_df = df[df["FINBERT_SCORE"] >= POS_THRESHOLD]
        pos_text = " ".join(pos_df["HEADLINE"].tolist())
        print(f"🙂 Very positive headlines rows: {len(pos_df)}")

        make_wordcloud(
            pos_text,
            f"Very Positive Headlines (FINBERT ≥ {POS_THRESHOLD})",
            OUTPUT_DIR / "wordcloud_positive_with_legend.png",
        )

        # ==================================================
        # 3) VERY NEGATIVE HEADLINES
        # ==================================================
        neg_df = df[df["FINBERT_SCORE"] <= NEG_THRESHOLD]
        neg_text = " ".join(neg_df["HEADLINE"].tolist())
        print(f"🙁 Very negative headlines rows: {len(neg_df)}")

        make_wordcloud(
            neg_text,
            f"Very Negative Headlines (FINBERT ≤ {NEG_THRESHOLD})",
            OUTPUT_DIR / "wordcloud_negative_with_legend.png",
        )
    else:
        print("⚠️ FINBERT_SCORE column not found – skipping pos/neg word clouds.")


if __name__ == "__main__":
    main()
