"""
Advanced training script (optional). Run only if you want to re-train or improve the model.

Outputs:
- /mnt/data/car_price_advanced.joblib
- /mnt/data/car_price_advanced_cv_results.json
- /mnt/data/shap_summary.png (if shap installed)
"""

import os
import json
import joblib
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

DATA_PATH = "car data.csv"
OUT_MODEL = "car_price_model.joblib"
OUT_CV_JSON = "car_price_advanced_cv_results.json"
OUT_SHAP = "shap_summary.png"

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)

# Detect target column
possible_targets = [c for c in df.columns if 'price' in c.lower()] or df.select_dtypes(include=[np.number]).columns.tolist()[-1:]
target = possible_targets[0]
print("Detected target:", target)

# Drop missing target rows
df = df.dropna(subset=[target]).reset_index(drop=True)

# Basic feature engineering
X = df.drop(columns=[target])
y = df[target].astype(float)

if 'Year' in X.columns or 'year' in [c.lower() for c in X.columns]:
    # create Car_Age if Year present
    year_candidates = [c for c in X.columns if c.lower() == 'year']
    if year_candidates:
        yc = year_candidates[0]
        X['Car_Age'] = (pd.Timestamp.now().year - X[yc]).astype(int)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Reduce high cardinality categories
for c in cat_cols:
    top = X[c].value_counts().nlargest(50).index
    X[c] = X[c].where(X[c].isin(top), other='Other')

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# Base learners
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
estimators = [('rf', rf)]

# Try to add xgboost / lightgbm if available
try:
    import xgboost as xgb
    xgbm = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=400, random_state=42, n_jobs=4)
    estimators.append(('xgb', xgbm))
except Exception:
    xgb = None

try:
    import lightgbm as lgb
    lgbm = lgb.LGBMRegressor(n_estimators=400, random_state=42)
    estimators.append(('lgb', lgbm))
except Exception:
    lgb = None

meta = Ridge(alpha=1.0)
stack = StackingRegressor(estimators=estimators, final_estimator=meta, cv=5, n_jobs=-1, passthrough=False)

pipeline = Pipeline([
    ('pre', preprocessor),
    ('stack', stack)
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_rmse(pipe, X, y, cv=kf):
    scores = -cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    return np.sqrt(scores)

print("Running CV evaluation ...")
# Optionally transform target (e.g., log1p) if very skewed. Here we use raw target for simplicity.
rmse_folds = cv_rmse(pipeline, X, y)
print("CV RMSE mean:", rmse_folds.mean(), "std:", rmse_folds.std())

print("Fitting full pipeline ...")
pipeline.fit(X, y)

print("Saving advanced pipeline to:", OUT_MODEL)
joblib.dump(pipeline, OUT_MODEL)

cv_summary = {
    'cv_rmse_mean': float(rmse_folds.mean()),
    'cv_rmse_std': float(rmse_folds.std()),
    'n_rows': int(X.shape[0]),
    'n_features': int(X.shape[1])
}
with open(OUT_CV_JSON, 'w') as f:
    json.dump(cv_summary, f, indent=2)
print("Saved CV summary to:", OUT_CV_JSON)

# SHAP optional (best-effort)
try:
    import shap
    import matplotlib.pyplot as plt
    print("Computing SHAP values (may take some time) ...")
    X_trans = pipeline.named_steps['pre'].transform(X)
    # pick first tree-based estimator for explainer if available
    base = None
    for name, est in pipeline.named_steps['stack'].estimators:
        base = est
        break
    if base is not None:
        explainer = shap.Explainer(base)
        shap_vals = explainer(X_trans)
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_vals, features=X_trans, show=False)
        plt.tight_layout()
        plt.savefig(OUT_SHAP, dpi=200)
        plt.close()
        print("Saved SHAP summary to:", OUT_SHAP)
except Exception as e:
    print("SHAP not available or failed:", e)

print("Advanced training finished.")
