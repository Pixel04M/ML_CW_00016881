
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- Page Config --------------------------
st.set_page_config(
    page_title="Vehicle Crash Injury Severity Predictor",
    page_icon="car",
    layout="wide"
)

st.title("Vehicle Crash Injury Severity Prediction")
st.markdown("""
This app predicts the **Injury Severity** in vehicle crashes using a pre-trained ML model  
(based on Montgomery County Crash Reporting - Drivers Data).
""")

# -------------------------- Load Model & Artifacts --------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    info = joblib.load("preprocessing_info.pkl")
    return model, scaler, info

try:
    model, scaler, info = load_artifacts()
    st.success("Model and preprocessing artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Make sure these files are in the same folder:\n"
            "- `best_model.pkl`\n"
            "- `scaler.pkl`\n"
            "- `preprocessing_info.pkl`")
    st.stop()

# Extract info
selected_features     = info['selected_features']
categorical_features  = info['categorical_features']
numerical_features    = info['numerical_features']
final_feature_names   = info['feature_names']
best_model_name       = info['best_model']

# -------------------------- Preprocessing Function --------------------------
def preprocess_user_input(user_df):
    """Replicate exactly the preprocessing done during training."""
    df = user_df.copy()

    # 1. Feature Engineering
    # Crash hour & day of week
    if 'Crash Date/Time' in df.columns:
        df['Crash Date/Time'] = pd.to_datetime(df['Crash Date/Time'], errors='coerce')
        df['Crash Hour'] = df['Crash Date/Time'].dt.hour
        df['Crash DayOfWeek'] = df['Crash Date/Time'].dt.dayofweek
    else:
        df['Crash Hour'] = np.nan
        df['Crash DayOfWeek'] = np.nan

    # Vehicle Age
    if 'Vehicle Year' in df.columns:
        current_year = 2025
        df['Vehicle Age'] = current_year - df['Vehicle Year'].replace({0: np.nan})
        median_age = df['Vehicle Age'].median()
        df['Vehicle Age'] = df['Vehicle Age'].fillna(median_age if pd.notna(median_age) else 10)

    # Substance abuse flag
    if 'Driver Substance Abuse' in df.columns:
        df['Has_Substance_Abuse'] = df['Driver Substance Abuse'].apply(
            lambda x: 1 if pd.notna(x) and ('ALCOHOL' in str(x).upper() or 
                                         'DRUG' in str(x).upper() or 
                                         'SUSPECTED' in str(x).upper()) else 0
        )

    # 2. Ensure all selected features exist
    for col in selected_features:
        if col not in df.columns:
            df[col] = np.nan

    X = df[selected_features].copy()

    # 3. Fill missing values (same as training)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown')
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)

    # 4. One-hot encoding
    X_encoded = pd.get_dummies(X[categorical_features], prefix=categorical_features, drop_first=True)

    # 5. Combine numerical + encoded
    X_final = pd.concat([X[numerical_features], X_encoded], axis=1)

    # 6. Add missing columns that model expects (set to 0)
    for col in final_feature_names:
        if col not in X_final.columns:
            X_final[col] = 0

    # 7. Reorder columns exactly as during training
    X_final = X_final.reindex(columns=final_feature_names, fill_value=0)

    # 8. Scale using the saved scaler
    X_scaled = scaler.transform(X_final)

    return X_scaled

# -------------------------- Sidebar Input --------------------------
st.sidebar.header("Enter Crash Details")

def get_user_input():
    data = {}

    data['Weather'] = st.sidebar.selectbox("Weather", 
        ['CLEAR', 'CLOUDY', 'RAINING', 'SNOWING', 'FOGGY', 'UNKNOWN', 'OTHER'])

    data['Surface Condition'] = st.sidebar.selectbox("Surface Condition",
        ['DRY', 'WET', 'SNOW', 'ICE', 'SLUSH', 'UNKNOWN'])

    data['Light'] = st.sidebar.selectbox("Light Conditions",
        ['DAYLIGHT', 'DARK LIGHTS ON', 'DUSK', 'DAWN', 'DARK NO LIGHTS', 'UNKNOWN'])

    data['Traffic Control'] = st.sidebar.selectbox("Traffic Control",
        ['NO CONTROLS', 'TRAFFIC SIGNAL', 'STOP SIGN', 'YIELD SIGN', 'UNKNOWN'])

    data['Collision Type'] = st.sidebar.selectbox("Collision Type",
        ['SINGLE VEHICLE', 'HEAD ON', 'REAR END', 'ANGLE', 'SIDESWIPE SAME DIRECTION', 'OTHER'])

    data['Driver At Fault'] = st.sidebar.selectbox("Driver At Fault?", ['Yes', 'No'])

    data['Driver Substance Abuse'] = st.sidebar.selectbox("Driver Substance Abuse",
        ['NONE', 'ALCOHOL PRESENT', 'ALCOHOL CONTRIBUTED', 'DRUGS', 'UNKNOWN'])

    data['Vehicle Damage Extent'] = st.sidebar.selectbox("Vehicle Damage Extent",
        ['NONE', 'MINOR', 'FUNCTIONAL', 'DISABLING', 'SUPERFICIAL', 'UNKNOWN'])

    data['Vehicle Body Type'] = st.sidebar.selectbox("Vehicle Body Type",
        ['PASSENGER CAR', 'SUV', 'PICKUP', 'VAN', 'MOTORCYCLE', 'BUS', 'TRUCK', 'OTHER'])

    data['Vehicle Movement'] = st.sidebar.selectbox("Vehicle Movement",
        ['MOVING', 'STOPPED', 'PARKED', 'SLOWING', 'UNKNOWN'])

    data['Speed Limit'] = st.sidebar.number_input("Speed Limit (mph)", 0, 100, 35)

    data['Vehicle Year'] = st.sidebar.number_input("Vehicle Year", 1900, 2025, 2018)

    data['Route Type'] = st.sidebar.selectbox("Route Type",
        ['County', 'State', 'Municipality', 'Interstate', 'US Route', 'Unknown'])

    # Optional datetime for hour/day extraction
    date_time = st.sidebar.text_input("Crash Date/Time (YYYY-MM-DD HH:MM)", "2025-01-01 12:00")
    data['Crash Date/Time'] = date_time

    return pd.DataFrame([data])

input_df = get_user_input()

# -------------------------- Prediction --------------------------
if st.sidebar.button("Predict Injury Severity"):
    with st.spinner("Making prediction..."):
        try:
            X_ready = preprocess_input(input_df)
            pred = model.predict(X_ready)[0]
            proba = model.predict_proba(X_ready)[0]

            classes = model.classes_
            proba_df = pd.DataFrame({
                'Injury Severity': classes,
                'Probability': proba
            }).sort_values('Probability', ascending=False).reset_index(drop=True)

            st.subheader("Prediction Result")
            st.success(f"**Predicted Injury Severity: {pred}**")

            st.subheader("Probability Distribution")
            st.dataframe(proba_df.style.format({"Probability": "{:.1%}"}))

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=proba_df, y='Injury Severity', x='Probability', palette="viridis", ax=ax)
            ax.set_title("Prediction Confidence")
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        except Exception as e:
            st.error("Error during prediction. Check input data.")
            st.exception(e)

# -------------------------- Footer --------------------------
st.sidebar.markdown("---")
st.sidebar.info(f"Model: **{best_model_name}**")

st.markdown("---")
st.caption("Model trained on Montgomery County, MD crash data (2020–2024). Predictions are probabilistic.")