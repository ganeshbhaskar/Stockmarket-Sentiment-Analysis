import os
import pandas as pd
import refinitiv.data as rd

# ---- LSEG news import (SDK-safe) ----
try:
    from refinitiv.data.content import news as rd_news
except Exception:
    import refinitiv.data.content.news as rd_news  # type: ignore

# ---- Config ----
OUTPUT_DIR = r"../data/raw"
RIC        = "AAPL.O"
LANG       = "EN"
MAX_ROWS   = 2000

# Date range (your requirement)
START_DATE = pd.Timestamp("2025-08-10")
END_DATE   = pd.Timestamp("2025-11-22")


def to_df(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "df"):
        return x.df
    if hasattr(x, "raw"):
        return pd.DataFrame(x.raw)
    try:
        return pd.DataFrame(x)
    except Exception:
        return pd.DataFrame()


def main():

    rd.open_session()
    print("✅ Session opened for news.")

    # ❗ QUERY WITHOUT DATE FILTER (This always works)
    query = f"R:{RIC} AND Language:{LANG}"
    print("\n🔍 Query Used:")
    print(query)

    # Fetch headlines (max rows)
    res = rd_news.headlines.Definition(
        query=query,
        count=MAX_ROWS
    ).get_data()

    df = to_df(res.data)

    if df.empty:
        print("⚠️ Still no data. Something is wrong in SDK or permissions.")
        return

    # Move timestamp index into column
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "TIMESTAMP"})
    else:
        df = df.reset_index().rename(columns={"index": "TIMESTAMP"})

    # Standardize columns
    rename_map = {
        "headline": "HEADLINE",
        "storyId": "STORY_ID",
        "sourceCode": "SOURCE",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Convert timestamp
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce", utc=True)
    df["DATE"] = df["TIMESTAMP"].dt.date

    # Drop bad rows
    df = df.dropna(subset=["TIMESTAMP"])

    # Apply date filter (THIS PART FIXES YOUR ISSUE)
    df = df[(df["DATE"] >= START_DATE.date()) & (df["DATE"] <= END_DATE.date())]

    print(f"\n📅 Filtered to range {START_DATE.date()} → {END_DATE.date()}")
    print(f"📰 Rows after filtering: {len(df)}")

    # Add RIC
    df["RIC"] = RIC

    # Keep only needed columns
    keep_cols = ["TIMESTAMP", "DATE", "HEADLINE", "SOURCE", "STORY_ID", "RIC"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Remove duplicate stories
    if "STORY_ID" in df.columns:
        df = df.drop_duplicates(subset=["STORY_ID"])

    df = df.sort_values("TIMESTAMP")

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"news_{RIC}.csv")
    df.to_csv(out_file, index=False)

    print(f"\n💾 Saved {len(df)} headlines → {out_file}")
    print("\n🔹 Sample:")
    print(df.head())


if __name__ == "__main__":
    main()
