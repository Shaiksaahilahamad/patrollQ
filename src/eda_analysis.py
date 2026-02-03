import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    INPUT_PATH = "data/processed/chicago_crime_cleaned.csv"
    Path("outputs").mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    # Crime Type Distribution
    crime_counts = df["Primary Type"].value_counts().head(10)
    crime_counts.plot(kind="bar", title="Top 10 Crime Types")
    plt.ylabel("Number of Crimes")
    plt.tight_layout()
    plt.savefig("outputs/eda_crime_types.png")
    plt.close()

    # Arrest vs Non-Arrest
    df["Arrest"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        title="Arrest vs Non-Arrest"
    )
    plt.ylabel("")
    plt.savefig("outputs/eda_arrest_rate.png")
    plt.close()

    # Domestic vs Non-Domestic
    df["Domestic"].value_counts().plot(
        kind="bar",
        title="Domestic vs Non-Domestic Crimes"
    )
    plt.ylabel("Count")
    plt.savefig("outputs/eda_domestic.png")
    plt.close()

    print("✅ EDA analysis completed")

if __name__ == "__main__":
    main()
