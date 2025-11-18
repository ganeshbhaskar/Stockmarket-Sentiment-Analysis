import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------- CONFIG ---------
MODEL_NAME   = "ProsusAI/finbert"
INPUT_CSV    = Path(__file__).resolve().parents[1] / "data" / "raw" / "news_AAPL.O.csv"
OUTPUT_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_CSV   = OUTPUT_DIR / "news_AAPL.O_finbert.csv"

HEADLINE_COL_CANDIDATES = ["HEADLINE", "headline", "Title", "title"]
BATCH_SIZE   = 16
MAX_LENGTH   = 128
# --------------------------


def find_headline_column(df: pd.DataFrame) -> str:
    """Try to locate the headline text column."""
    for col in HEADLINE_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a headline column. "
        f"Looked for {HEADLINE_COL_CANDIDATES}, got columns: {list(df.columns)}"
    )


def load_finbert():
    """Load tokenizer & model, move model to CPU/GPU."""
    print(f"📥 Loading FinBERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # label mapping e.g. {0: 'negative', 1: 'neutral', 2: 'positive'}
    id2label = model.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}

    pos_idx = label2id["positive"]
    neg_idx = label2id["negative"]
    neu_idx = label2id["neutral"]

    print(f"🔤 FinBERT label mapping: {id2label}")
    return tokenizer, model, device, pos_idx, neg_idx, neu_idx


def chunked(iterable, size):
    """Yield successive chunks from a list."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def score_with_finbert(texts, tokenizer, model, device, pos_idx, neg_idx, neu_idx):
    """
    Run FinBERT on a list of texts.
    Returns three lists: positive_probs, negative_probs, neutral_probs.
    """
    all_pos, all_neg, all_neu = [], [], []

    for batch in chunked(texts, BATCH_SIZE):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        all_pos.extend(probs[:, pos_idx].tolist())
        all_neg.extend(probs[:, neg_idx].tolist())
        all_neu.extend(probs[:, neu_idx].tolist())

    return all_pos, all_neg, all_neu


def add_date_from_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure every row has a valid DATE derived from TIMESTAMP.
    - Parses TIMESTAMP to datetime (UTC).
    - Drops rows where parsing fails.
    - Creates DATE in 'dd/mm/yyyy' string format for the CSV.
    """
    if "TIMESTAMP" not in df.columns:
        raise KeyError("TIMESTAMP column not found in news file; cannot derive DATE.")

    print("⏱ Parsing TIMESTAMP → DATE ...")
    ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce", utc=True)

    df["TIMESTAMP_PARSED"] = ts  # keep parsed version just in case
    df["DATE"] = ts.dt.date      # pure date (no time)

    # Count and drop rows with invalid timestamps
    bad = df["DATE"].isna().sum()
    if bad > 0:
        print(f"⚠️ Dropping {bad} rows with invalid TIMESTAMP.")
        df = df.dropna(subset=["DATE"]).reset_index(drop=True)

    # Format as dd/mm/yyyy for the output CSV (nice for Excel)
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%d/%m/%Y")

    return df


def main():
    # ---------- Load CSV ----------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found at: {INPUT_CSV}")

    print(f"📄 Reading news from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Identify headline column
    headline_col = find_headline_column(df)
    print(f"📰 Using '{headline_col}' as headline text column.")

    # Drop rows with empty headlines
    df = df.dropna(subset=[headline_col]).reset_index(drop=True)
    if df.empty:
        print("⚠️ No headlines found after dropping empty rows.")
        return

    # ---------- Ensure DATE from TIMESTAMP ----------
    df = add_date_from_timestamp(df)

    # ---------- Load FinBERT ----------
    tokenizer, model, device, pos_idx, neg_idx, neu_idx = load_finbert()

    # ---------- Run sentiment ----------
    headlines = df[headline_col].astype(str).tolist()
    print(f"🔎 Scoring {len(headlines)} headlines with FinBERT...")

    pos, neg, neu = score_with_finbert(
        headlines, tokenizer, model, device, pos_idx, neg_idx, neu_idx
    )

    # Add sentiment columns
    df["FINBERT_POS"] = pos
    df["FINBERT_NEG"] = neg
    df["FINBERT_NEU"] = neu

    # Continuous score: POS - NEG  (≈ [-1, 1])
    df["FINBERT_SCORE"] = df["FINBERT_POS"] - df["FINBERT_NEG"]

    # Discrete label
    df["FINBERT_LABEL"] = df["FINBERT_SCORE"].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )

    # ---------- Save ----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Done. Saved {len(df):,} rows → {OUTPUT_CSV}")
    print("📑 Columns now include: DATE (dd/mm/yyyy), "
          "FINBERT_POS, FINBERT_NEG, FINBERT_NEU, FINBERT_SCORE, FINBERT_LABEL")
    print(df[[headline_col, "DATE", "FINBERT_SCORE", "FINBERT_LABEL"]].head())


if __name__ == "__main__":
    main()
