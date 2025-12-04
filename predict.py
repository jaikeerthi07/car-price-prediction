import joblib
import pandas as pd
import numpy as np
import os


_cached_model = None


def load_model(path):
    """Load and return model pipeline from disk. Caches the loaded object."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        _cached_model = joblib.load(path)
    return _cached_model


def predict_df(model, df: pd.DataFrame) -> pd.Series:
    """
    Run prediction on a DataFrame and return a pandas Series of predictions.
    This function will try to align columns — but the safest option is to pass a DataFrame
    with the exact same columns (and types) that were used during training.
    """
    if df.shape[0] == 0:
        return pd.Series([], dtype=float)


    # If training pipeline expects extra columns and a ColumnTransformer with drop remainder,
    # predictions will fail if columns are missing. We simply pass the DataFrame through the pipeline.
    preds = model.predict(df)
    return pd.Series(preds, index=df.index)


if __name__ == '__main__':
    # small smoke test
    MODEL_PATH = "car_price_model.joblib"
    m = load_model(MODEL_PATH)
    print('Loaded model:', m)