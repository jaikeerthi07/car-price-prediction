"""
Streamlit app for Car Price Prediction.

- Loads a trained scikit-learn pipeline from: /mnt/data/car_price_model.joblib
- Allows single-row prediction (paste header row OR use a generic form) and batch CSV upload.
- Avoids the earlier IndentationError by using proper blocks and guards.
- NOTE: If your model expects specific columns, ensure the form fields / uploaded CSV match those columns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from predict import load_model, predict_df

# Paths in this environment:
MODEL_PATH = "car_price_model.joblib"     # saved pipeline from previous run
DATASET_PATH = "car data.csv"            # original uploaded dataset path (if needed)

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("Car Price Prediction")

st.markdown(
    "This demo loads a trained scikit-learn pipeline and predicts car selling prices.\n\n"
    f"Model path used: `{MODEL_PATH}`\n\n"
    f"Dataset path (if you need it): `{DATASET_PATH}`"
)

@st.cache_resource
def _load():
    return load_model(MODEL_PATH)

try:
    model = _load()
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}: {e}")
    st.stop()

st.sidebar.header("Prediction options")
mode = st.sidebar.radio("Mode", ["Single prediction (form)", "Batch prediction (CSV upload)"])

if mode == "Single prediction (form)":
    st.subheader("Enter car details")
    st.info("You can either paste a CSV header row to generate form fields, or use the generic form.")

    # Option A — header-driven form
    st.markdown("**Option A — Paste a CSV header row (comma-separated)**")
    header_row = st.text_input("Paste header row (comma-separated). Leave blank to show a generic input.")

    sample = {}
    if header_row is not None and header_row.strip():
        # Guaranteed indented block under the `if` — fixes the IndentationError
        cols = [c.strip() for c in header_row.split(',') if c.strip()]
        st.write(f"Detected columns: {cols}")

        # Build inputs for each column in the pasted header.
        # We'll guess field types simply: if column name suggests numeric use number_input, else text_input.
        for c in cols:
            lname = c.lower()
            if any(tok in lname for tok in ['year', 'kms', 'km', 'driven', 'price', 'present', 'engine', 'power']):
                # numeric field
                sample[c] = st.text_input(c, value="")
            else:
                sample[c] = st.text_input(c, value="")

        if st.button("Predict from form with header"):
            try:
                # Convert numeric-like inputs to numbers where possible
                df_row = {}
                for k, v in sample.items():
                    v_str = v.strip()
                    if v_str == "":
                        df_row[k] = None
                    else:
                        # try to parse numeric
                        try:
                            if '.' in v_str:
                                df_row[k] = float(v_str)
                            else:
                                df_row[k] = int(v_str)
                        except Exception:
                            df_row[k] = v_str
                df = pd.DataFrame([df_row])
                preds = predict_df(model, df)
                st.success(f"Predicted selling price: {preds.iloc[0]:.4f}")
                st.write(df.assign(predicted_price=preds))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        # Option B — Generic common fields (adjust as needed for your dataset)
        st.markdown("**Option B — Generic common fields**")
        Car_Name = st.text_input('Car_Name', value='city')
        Year = st.number_input('Year', min_value=1900, max_value=2100, value=2014)
        Driven_kms = st.number_input('Driven_kms', min_value=0, value=50000)
        Fuel_Type = st.selectbox('Fuel_Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Other'])
        Selling_type = st.selectbox('Selling_type', ['Individual', 'Dealer'])
        Transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
        Owner = st.selectbox('Owner', ['First', 'Second', 'Third', 'Fourth & Above'])
        Present_Price = st.number_input('Present_Price', min_value=0.0, value=5.0, format="%f")

        owner_mapping = {'First': 0, 'Second': 1, 'Third': 3, 'Fourth & Above': 3}
        Owner_val = owner_mapping[Owner]

        if st.button('Predict (generic)'):
            Car_Age = pd.Timestamp.now().year - Year
            row = {
                'Car_Name': Car_Name,
                'Year': Year,
                'Driven_kms': Driven_kms,
                'Fuel_Type': Fuel_Type,
                'Selling_type': Selling_type,
                'Transmission': Transmission,
                'Owner': Owner_val,
                'Present_Price': Present_Price,
                'Car_Age': Car_Age
            }
            df = pd.DataFrame([row])
            try:
                preds = predict_df(model, df)
                st.success(f"Predicted selling price: {preds.iloc[0]:.4f}")
                st.write(df.assign(predicted_price=preds))
            except Exception as e:
                st.error(f"Prediction failed: {e}")


elif mode == "Batch prediction (CSV upload)":
    st.subheader("Upload CSV for batch prediction")
    st.markdown(
        "Upload a CSV where each row is a car record with the same columns used during training (excluding the target).\n"
        "Example columns: Year, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner, Present_Price, etc."
    )
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded file preview:")
            st.dataframe(df.head())
            
            if st.button('Run batch prediction'):
                # Preprocessing for batch prediction
                
                # 1. Calculate Car_Age
                if 'Car_Age' not in df.columns and 'Year' in df.columns:
                    df['Car_Age'] = pd.Timestamp.now().year - df['Year']
                
                # 2. Map Owner if string
                if df['Owner'].dtype == 'object':
                    owner_mapping = {'First': 0, 'Second': 1, 'Third': 3, 'Fourth & Above': 3}
                    # Use map and fillna with 0 or keep original if numeric-like
                    df['Owner'] = df['Owner'].map(owner_mapping).fillna(0)
                
                # 3. Ensure Car_Name exists (fill with 'city' if missing, though model might not use it heavily if high cardinality was reduced)
                if 'Car_Name' not in df.columns:
                    df['Car_Name'] = 'city'

                preds = predict_df(model, df)
                out = df.copy()
                out['predicted_selling_price'] = preds
                st.success('Predictions complete')
                st.dataframe(out.head(50))
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions (CSV)', csv, 'predictions.csv', 'text/csv')

                # --- Advanced Visualizations ---
                st.markdown("---")
                st.subheader("Prediction Analysis")

                # 1. Distribution of Predicted Prices
                fig_hist = px.histogram(out, x='predicted_selling_price', nbins=30, title="Distribution of Predicted Selling Prices")
                st.plotly_chart(fig_hist, use_container_width=True)

                # 2. Actual vs Predicted (if Selling_Price exists)
                if 'Selling_Price' in out.columns:
                    fig_scatter = px.scatter(out, x='Selling_Price', y='predicted_selling_price', 
                                             title="Actual vs Predicted Selling Price",
                                             labels={'Selling_Price': 'Actual Price', 'predicted_selling_price': 'Predicted Price'},
                                             trendline="ols")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 3. Feature vs Predicted Price (Interactive)
                st.markdown("### Feature vs Predicted Price")
                feature_to_plot = st.selectbox("Select feature to plot against predicted price", 
                                               ['Year', 'Driven_kms', 'Present_Price', 'Car_Age'])
                
                if feature_to_plot in out.columns:
                    fig_feat = px.scatter(out, x=feature_to_plot, y='predicted_selling_price', 
                                          color='Fuel_Type' if 'Fuel_Type' in out.columns else None,
                                          title=f"{feature_to_plot} vs Predicted Selling Price")
                    st.plotly_chart(fig_feat, use_container_width=True)

                # 4. Correlation Heatmap (Numeric columns only)
                st.markdown("### Correlation Heatmap")
                numeric_df = out.select_dtypes(include=['float64', 'int64'])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
                    st.plotly_chart(fig_corr, use_container_width=True)

                # 5. Box Plot of Price by Fuel Type
                if 'Fuel_Type' in out.columns:
                    st.markdown("### Price Distribution by Fuel Type")
                    fig_box = px.box(out, x='Fuel_Type', y='predicted_selling_price', points="all",
                                     title="Predicted Price Distribution by Fuel Type")
                    st.plotly_chart(fig_box, use_container_width=True)

        except Exception as e:
            st.error(f"Could not read uploaded CSV or run predictions: {e}")

st.markdown("---")
st.caption("If prediction fails, verify the input column names match those used in training.")
