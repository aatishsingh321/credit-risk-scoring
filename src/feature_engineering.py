import pandas as pd

def preprocess(df):
    """
    Clean data, create target variable, and encode features
    """

    # Keep only useful loan statuses
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

    # Target variable
    df["default_flag"] = df["loan_status"].apply(
        lambda x: 1 if x == "Charged Off" else 0
    )

    # Select important features
    df = df[
        [
            "loan_amnt",
            "annual_inc",
            "term",
            "grade",
            "emp_length",
            "default_flag"
        ]
    ]

    # Handle missing values
    df["annual_inc"] = df["annual_inc"].fillna(df["annual_inc"].median())
    df["emp_length"] = df["emp_length"].fillna("Unknown")

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df
