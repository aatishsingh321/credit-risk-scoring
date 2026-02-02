import pandas as pd
from config import RAW_DATA_PATH

def load_data():
    """
    Load loan dataset from raw data folder
    """
    df = pd.read_csv(RAW_DATA_PATH + "loan.csv")
    return df
