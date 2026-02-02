from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from config import TARGET_COL, TEST_SIZE, RANDOM_STATE

def train_model(df):
    """
    Train LightGBM model for credit risk prediction
    """

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
