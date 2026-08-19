from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TRUE_FILE = DATA_DIR / "True.csv"
FAKE_FILE = DATA_DIR / "Fake.csv"

OUTPUT_FILE = PROCESSED_DIR / "cleaned_dataset.csv"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    true_df = pd.read_csv(TRUE_FILE)
    fake_df = pd.read_csv(FAKE_FILE)

    return true_df, fake_df


def clean_dataframe(
    df: pd.DataFrame,
    label: int
) -> pd.DataFrame:

    df = df.copy()

    # Add classification label
    df["label"] = label

    # Ensure text columns are strings
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Remove leading/trailing whitespace
    df["title"] = df["title"].str.strip()
    df["text"] = df["text"].str.strip()

    # Remove rows where both title and text are empty
    df = df[
        (df["title"] != "") |
        (df["text"] != "")
    ]

    return df


def create_combined_dataset() -> pd.DataFrame:

    true_df, fake_df = load_datasets()

    # 1 = Genuine
    true_df = clean_dataframe(true_df, label=1)

    # 0 = Fake
    fake_df = clean_dataframe(fake_df, label=0)

    # Combine datasets
    combined_df = pd.concat(
        [true_df, fake_df],
        ignore_index=True
    )

    # Remove exact duplicate rows
    combined_df = combined_df.drop_duplicates()

    # Create model input text
    combined_df["model_text"] = (
        combined_df["title"] + " " + combined_df["text"]
    ).str.strip()

    # Remove rows with empty model input
    combined_df = combined_df[
        combined_df["model_text"] != ""
    ]

    return combined_df


def main() -> None:

    print("=" * 60)
    print("CREATING CLEAN DATASET")
    print("=" * 60)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = create_combined_dataset()

    print(f"Final rows: {len(df)}")

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "Fake",
                1: "Genuine"
            }
        )
    )

    # Save only required columns
    output_df = df[
        [
            "title",
            "text",
            "subject",
            "date",
            "model_text",
            "label"
        ]
    ]

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved dataset:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()