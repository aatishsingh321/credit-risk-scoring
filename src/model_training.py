from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, TEST_SIZE, RANDOM_STATE

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
    n_estimators=400,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=31,
    min_child_samples=50,
    class_weight="balanced",
    random_state=RANDOM_STATE
)


    model.fit(X_train, y_train)

    return model, X_test, y_test
