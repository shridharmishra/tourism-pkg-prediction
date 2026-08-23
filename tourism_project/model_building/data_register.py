import os

import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "Occupation",
    "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched",
    "NumberOfFollowups", "DurationOfPitch",
]


def register_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Upload tourism.csv into tourism_project/data/ first."
        )

    df = pd.read_csv(data_path)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    print("Dataset registered successfully.")
    print(f"Path         : {data_path}")
    print(f"Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns      : {list(df.columns)}")
    print(f"Missing vals : {int(df.isnull().sum().sum())} total")
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts())

    return df


if __name__ == "__main__":
    register_dataset()
