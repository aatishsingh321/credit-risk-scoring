from src.data_ingestion import load_data
from src.data_validation import validate_data
from src.feature_engineering import preprocess
from src.model_training import train_model
from src.model_evaluation import evaluate

print("Loading data...")
df = load_data()

print("Validating data...")
validate_data(df)

print("Preprocessing data...")
df = preprocess(df)

print("Training model...")
model, X_test, y_test = train_model(df)

print("Evaluating model...")
evaluate(model, X_test, y_test)
