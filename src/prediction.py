import joblib
import pandas as pd

from src.preprocessing import prepare_features

def load_model_bundle(model_path):
    """Load the saved model bundle."""
    return joblib.load(model_path)

def transform_features(data: pd.DataFrame, model_bundle):
    features = prepare_features(data)
    return model_bundle["preprocessor"].transform(features)

def predict(data: pd.DataFrame, model_bundle):
    """
    Generate fraud probabilities and predictions for transaction data.
    """
    features = prepare_features(data)

    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]
    threshold = model_bundle["threshold"]

    transformed_features = preprocessor.transform(features)
    probabilities = model.predict_proba(transformed_features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return probabilities, predictions
