import pandas as pd
from predict import load_model, predict_df

MODEL_PATH = "car_price_model.joblib"
model = load_model(MODEL_PATH)

# Simulate uploaded CSV (missing Car_Age, string Owner)
data = {
    'Car_Name': ['city', 'swift'],
    'Year': [2014, 2015],
    'Driven_kms': [50000, 40000],
    'Fuel_Type': ['Petrol', 'Diesel'],
    'Selling_type': ['Individual', 'Dealer'],
    'Transmission': ['Manual', 'Manual'],
    'Owner': ['First', 'Second'],
    'Present_Price': [5.0, 6.0],
    'Selling_Price': [3.0, 4.0] # Optional, for visualization
}
df = pd.DataFrame(data)
print("Input DataFrame (Simulated Upload):")
print(df)

# --- Logic from app.py ---
print("\nApplying preprocessing...")
# 1. Calculate Car_Age
if 'Car_Age' not in df.columns and 'Year' in df.columns:
    df['Car_Age'] = pd.Timestamp.now().year - df['Year']

# 2. Map Owner if string
if df['Owner'].dtype == 'object':
    owner_mapping = {'First': 0, 'Second': 1, 'Third': 3, 'Fourth & Above': 3}
    df['Owner'] = df['Owner'].map(owner_mapping).fillna(0)

# 3. Ensure Car_Name exists
if 'Car_Name' not in df.columns:
    df['Car_Name'] = 'city'
# -------------------------

print("\nProcessed DataFrame:")
print(df)

try:
    preds = predict_df(model, df)
    print("\nBatch Prediction successful:")
    print(preds)
except Exception as e:
    print("\nBatch Prediction failed:")
    print(e)
