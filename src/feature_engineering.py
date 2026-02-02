import pandas as pd

def preprocess(df):
    """
    Strong feature engineering using credit bureau + financial behavior variables
    """

    # Keep only relevant loan outcomes
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

    # Target variable
    df["default_flag"] = df["loan_status"].apply(
        lambda x: 1 if x == "Charged Off" else 0
    )

    # Select HIGH-IMPACT FEATURES
    df = df[
        [
            "loan_amnt",
            "annual_inc",
            "installment",
            "dti",
            "revol_util",
            "inq_last_6mths",
            "delinq_2yrs",
            "open_acc",
            "total_acc",
            "term",
            "grade",
            "emp_length",
            "default_flag"
        ]
    ]

    # -------------------------
    # HANDLE MISSING VALUES
    # -------------------------
    num_cols = [
        "annual_inc", "installment", "dti", "revol_util",
        "inq_last_6mths", "delinq_2yrs", "open_acc", "total_acc"
    ]

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    df["emp_length"] = df["emp_length"].fillna("Unknown")

    # -------------------------
    # STRONG RATIO FEATURES
    # -------------------------
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df["installment_to_income"] = df["installment"] / df["annual_inc"]

    # -------------------------
    # ORDINAL GRADE ENCODING
    # -------------------------
    grade_map = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7}
    df["grade_num"] = df["grade"].map(grade_map)
    df.drop(columns=["grade"], inplace=True)

    # -------------------------
    # ENCODE CATEGORICAL FEATURES
    # -------------------------
    df = pd.get_dummies(df, columns=["term", "emp_length"], drop_first=True)

    return df
