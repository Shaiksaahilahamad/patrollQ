import pandas as pd
from pathlib import Path


def main():
    INPUT_PATH = "data/sampled/chicago_crime_500k.csv"
    OUTPUT_PATH = "data/processed/chicago_crime_cleaned.csv"

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("📥 Loading sampled dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Initial shape: {df.shape}")

    # -----------------------------
    # Date cleaning
    # -----------------------------
    print("🧹 Cleaning Date column...")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # -----------------------------
    # Remove duplicates
    # -----------------------------
    print("🧹 Removing duplicate crime IDs...")
    df = df.drop_duplicates(subset="ID")

    # -----------------------------
    # Geographic cleaning
    # -----------------------------
    print("🌍 Cleaning latitude & longitude...")
    df = df.dropna(subset=["Latitude", "Longitude"])

    df = df[
        (df["Latitude"].between(41.6, 42.1)) &
        (df["Longitude"].between(-88.0, -87.4))
    ]

    # -----------------------------
    # Boolean columns
    # -----------------------------
    for col in ["Arrest", "Domestic"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # -----------------------------
    # Categorical columns
    # -----------------------------
    categorical_cols = [
        "Primary Type",
        "Description",
        "Location Description",
        "Block"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    print(f"✅ Final cleaned shape: {df.shape}")

    # Save cleaned data
    df.to_csv(OUTPUT_PATH, index=False)
    print("💾 Cleaned data saved successfully")
    print(f"📁 Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
