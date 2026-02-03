import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw Chicago crime dataset.
    """
    print("📥 Loading raw dataset...")
    df = pd.read_csv(path)
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def sample_recent_records(df: pd.DataFrame, sample_size: int = 500_000) -> pd.DataFrame:
    """
    Sample the most recent crime records based on Date.
    """
    print("🧹 Converting Date column to datetime...")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    print("📅 Sorting records by most recent date...")
    df = df.sort_values("Date", ascending=False)

    print(f"🎯 Sampling latest {sample_size} records...")
    sampled_df = df.head(sample_size).reset_index(drop=True)

    print(f"✅ Sampled dataset shape: {sampled_df.shape}")
    return sampled_df


def main():
    RAW_DATA_PATH = "data/raw/chicago_crime_raw.csv"
    OUTPUT_PATH = "data/sampled/chicago_crime_500k.csv"

    Path("data/sampled").mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(RAW_DATA_PATH)
    df_sampled = sample_recent_records(df_raw)

    print("💾 Saving sampled dataset...")
    df_sampled.to_csv(OUTPUT_PATH, index=False)

    print("🎉 Data ingestion completed successfully!")
    print(f"📁 Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
