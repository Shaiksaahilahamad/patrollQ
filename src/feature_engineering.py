import pandas as pd
from pathlib import Path


def assign_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def main():
    INPUT_PATH = "data/processed/chicago_crime_cleaned.csv"
    OUTPUT_PATH = "data/processed/chicago_crime_features.csv"

    print("📥 Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    print(f"Initial shape: {df.shape}")

    # -----------------------------
    # Temporal Features
    # -----------------------------
    print("⏱️ Creating temporal features...")
    df["Hour"] = df["Date"].dt.hour
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Is_Weekend"] = (df["Date"].dt.weekday >= 5).astype(int)
    df["Season"] = df["Month"].apply(assign_season)

    # -----------------------------
    # Crime Severity Score
    # -----------------------------
    print("⚠️ Assigning crime severity scores...")
    severity_map = {
        "HOMICIDE": 5,
        "KIDNAPPING": 5,
        "CRIM SEXUAL ASSAULT": 5,
        "ROBBERY": 4,
        "ASSAULT": 4,
        "BATTERY": 4,
        "BURGLARY": 3,
        "MOTOR VEHICLE THEFT": 3,
        "THEFT": 2,
        "CRIMINAL DAMAGE": 2
    }

    df["Crime_Severity_Score"] = (
        df["Primary Type"]
        .str.upper()
        .map(severity_map)
        .fillna(1)
        .astype(int)
    )

    print(f"✅ Feature engineered shape: {df.shape}")

    # Save output
    Path("data/processed").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("💾 Feature engineering completed successfully")
    print(f"📁 Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
