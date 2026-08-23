import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Columns that are identifiers / row artifacts and carry no predictive signal
DROP_COLUMNS = ["CustomerID", "Unnamed: 0"]


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fix a known data-entry typo in the raw file ("Fe Male" -> "Female")
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    return df


def main():
    df = load_data()
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain: {Xtrain.shape}  Xtest: {Xtest.shape}")
    print(f"ytrain: {ytrain.shape}  ytest: {ytest.shape}")


if __name__ == "__main__":
    main()
