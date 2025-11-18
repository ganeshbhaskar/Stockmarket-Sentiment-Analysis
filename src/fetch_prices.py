import os
import pandas as pd
import refinitiv.data as rd
from refinitiv.data.content import historical_pricing as hp

OUTPUT_DIR = r"../data/raw"
RIC, START, END = "AAPL.O", "2025-08-01", "2025-11-30"

def to_dataframe(data_obj) -> pd.DataFrame:
    if isinstance(data_obj, pd.DataFrame):
        return data_obj
    if hasattr(data_obj, "df"):
        return data_obj.df
    if hasattr(data_obj, "raw"):
        return pd.DataFrame(data_obj.raw)
    try:
        return pd.DataFrame(data_obj)
    except Exception:
        return pd.DataFrame()

def main():
    rd.open_session()
    print("✅ Session opened.")

    res = hp.summaries.Definition(
        universe=RIC,
        interval="P1D",
        start=START,
        end=END
    ).get_data()

    df = to_dataframe(res.data)

    # ✅ Add TIMESTAMP column if the date is in the index
    if df.index.name is not None:
        df = df.reset_index().rename(columns={df.index.name: "TIMESTAMP"})
    elif "DATE" in df.columns:
        df = df.rename(columns={"DATE": "TIMESTAMP"})

    if df is None or len(df) == 0:
        print("⚠️ No rows returned.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"prices_{RIC}.csv")
    df.to_csv(out, index=False)

    print("📁 Saved →", out)
    print("🧭 Columns:", list(df.columns))

if __name__ == "__main__":
    main()
