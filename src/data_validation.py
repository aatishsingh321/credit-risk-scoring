def validate_data(df):
    """
    Perform basic data validation checks
    """
    print("==== DATA VALIDATION REPORT ====")
    print("\nMissing values per column:\n")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nData types:\n")
    print(df.dtypes)
