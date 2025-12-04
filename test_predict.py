import pandas as pd
from predict import load_model, predict_df

MODEL_PATH = "car_price_model.joblib"
model = load_model(MODEL_PATH)

# Test case with corrected column names
row = {
    'Car_Name': 'city',
    'Year': 2014,
    'Driven_kms': 50000,
    'Fuel_Type': 'Petrol',
    'Selling_type': 'Individual',
    'Transmission': 'Manual',
    'Owner': 0,
    'Present_Price': 5.0,
    'Car_Age': pd.Timestamp.now().year - 2014
}

df = pd.DataFrame([row])
print("Input DataFrame:")
print(df)

try:
    preds = predict_df(model, df)
    print("\nPrediction successful:")
    print(preds)
except Exception as e:
    print("\nPrediction failed:")
    print(e)
