from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# ----------------- CONFIG -----------------
ROOT = Path(__file__).resolve().parents[1]

# Use whichever file has your HEADLINE + DATE columns
NEWS_FILE   = ROOT / "data" / "raw" / "news_AAPL.O.csv"
OUTPUT_DIR  = ROOT / "data" / "processed" / "wordclouds"

# Example: all headlines (you can later filter to spike dates)
START_DATE = "2025-08-22"
END_DATE   = "2025-11-15"
# ------------------------------------------


def make_wordcloud(text: str, title: str, out_path: Path):
    """Generate a wordcloud + legend of top words and save it."""
    if not isinstance(text, str) or not text.strip():
        print(f"⚠️ Empty text for '{title}', skipping word cloud.")
        return

    # ---- stopwords ---------------------------------------------------------
    custom_stopwords = STOPWORDS | {
        "Apple", "AAPL", "Inc", "says", "said", "Reuters",
        "NS", "RTS", "RTTRS", "HINDU", "BUSST", "ZACKS", "CNBC"
    }

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=custom_stopwords,
        collocations=True,
    ).generate(text)

    # frequencies (0–1 normalized)
    word_freq = wc.words_                      # dict: word -> freq
    top_words = sorted(word_freq.items(),
                       key=lambda x: x[1],
                       reverse=True)[:20]

    # ---- figure with 2 panels: cloud + legend ------------------------------
    fig, (ax_wc, ax_leg) = plt.subplots(
        1, 2,
        figsize=(16, 9),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # word cloud
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title(title, fontsize=18)

    # legend as horizontal bar chart of top words
    if top_words:
        words, freqs = zip(*top_words)
        ax_leg.barh(range(len(words)), freqs, color="tabpurple", alpha=0.8)
        ax_leg.set_yticks(range(len(words)))
        ax_leg.set_yticklabels(words)
        ax_leg.invert_yaxis()
        ax_leg.set_xlabel("Normalized Frequency")
        ax_leg.set_title("Top Words (Legend)")
    else:
        ax_leg.text(
            0.5, 0.5, "No words",
            ha="center", va="center", fontsize=12
        )
        ax_leg.axis("off")

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼 Word cloud with legend saved → {out_path}")


def main():
    print(f"📄 Loading news → {NEWS_FILE}")
    df = pd.read_csv(NEWS_FILE)

    # adjust these column names if needed
    if "HEADLINE" not in df.columns:
        raise KeyError("Expected a 'HEADLINE' column in news file.")
    if "DATE" not in df.columns:
        raise KeyError("Expected a 'DATE' column for filtering.")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    mask = (df["DATE"] >= START_DATE) & (df["DATE"] <= END_DATE)
    df = df.loc[mask].dropna(subset=["HEADLINE"])

    print(f"✅ Headlines in range: {len(df)}")

    all_text = " ".join(df["HEADLINE"].astype(str).tolist())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "wordcloud_all_with_legend.png"
    make_wordcloud(all_text, "All Headlines (with Legend)", out_path)


if __name__ == "__main__":
    main()
