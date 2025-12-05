"""
Streamlit Multi-Page Application for Crash Reporting Analysis
Used for ML coursework
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set up the Streamlit page (title, icon, layout).
st.set_page_config(
    page_title="Crash Reporting Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic custom CSS for nicer page headings.
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Keep data, model, and scaler stored while switching pages.
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Sidebar navigation menu.
st.sidebar.title("🚗 Crash Analysis Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data Exploration", "🔮 Prediction"]
)

# ---------------------- HOME PAGE -------------------------
if page == "🏠 Home":
    # Title banner
    st.markdown('<div class="main-header">🚗 Crash Reporting Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    ## Welcome to the Crash Reporting Analysis Application
    
    This tool lets you explore crash data and build models to predict injury severity.
    
    **How to use this app:**
    1. Go to **Data Exploration** to view the dataset  
    2. Use **Prediction** to make predictions on new crash data  
    """)
    
    # Load dataset button
    if st.button("Load Dataset", type="primary"):
        try:
            df = pd.read_csv('Crash_Reporting_-_Drivers_Data.csv')
            st.session_state.data = df
            st.success(f" Dataset loaded! Shape: {df.shape}")
        except FileNotFoundError:
            st.error(" Dataset file not found. Make sure the CSV is in the same folder.")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

# ------------------ DATA EXPLORATION PAGE -----------------
elif page == "📊 Data Exploration":
    st.title(" Data Exploration")

    # Check if data is loaded
    if st.session_state.data is None:
        st.warning(" Please load the dataset from the Home page first.")
        
        # Quick load option
        if st.button("Load Dataset Now"):
            try:
                df = pd.read_csv('Crash_Reporting_-_Drivers_Data.csv')
                st.session_state.data = df
                st.success(" Dataset loaded! Refresh the page to see the data.")
            except Exception as e:
                st.error(f" Could not load dataset: {str(e)}")
    else:
        df = st.session_state.data
        
        # Show basic dataset stats
        st.header("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        col4.metric("Duplicate Rows", df.duplicated().sum())
        
        # Show a sample of the dataset
        st.subheader("Data Preview")
        num_rows = st.slider("Number of rows to display", 5, 100, 10)
        st.dataframe(df.head(num_rows))
        
        # Show summary statistics for numeric columns
        st.subheader("Summary Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        
        # Charts and graphs
        st.subheader("Visualizations")
        
        # Plot injury severity distribution
        if 'Injury Severity' in df.columns:
            st.write("### Injury Severity Distribution")
            injury_counts = df['Injury Severity'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            injury_counts.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Injury Severity')
            ax.set_xlabel('Injury Severity')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Close figure to prevent memory issues

# ---------------------- PREDICTION PAGE --------------------
elif page == "🔮 Prediction":
    st.title("🔮 Make Predictions")
    
    # Ensure model is loaded
    if st.session_state.model is None:
        st.warning("⚠️ Model needs to be loaded. Click the button below.")
        if st.button("Load Model Now"):
            try:
                model = joblib.load('best_model.pkl')
                st.session_state.model = model
                
                # Load scaler
                try:
                    scaler = joblib.load('scaler.pkl')
                    st.session_state.scaler = scaler
                except:
                    pass
                
                st.success("✅ Model loaded successfully! Refresh the page to use it.")
            except Exception as e:
                st.error(f"❌ Model file not found: {str(e)}. Train models in the notebook first.")
    else:
        st.header("Predict Injury Severity")
        st.subheader("Enter Crash Details")
        
        # Split form inputs into two columns
        col1, col2 = st.columns(2)
        
        # Left column inputs
        with col1:
            weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Snow", "Other"])
            surface = st.selectbox("Surface Condition", ["Dry", "Wet", "Snow", "Ice", "Other"])
            light = st.selectbox("Light", ["Daylight", "Dark - Lighted", "Dark - Not Lighted", "Dusk", "Dawn"])
            collision_type = st.selectbox("Collision Type", ["Front to Rear", "Front to Front", "Angle", "Sideswipe, Same Direction", "Single Vehicle", "Other"])
        
        # Right column inputs
        with col2:
            driver_at_fault = st.selectbox("Driver At Fault", ["Yes", "No"])
            speed_limit = st.slider("Speed Limit", 0, 100, 40)
            vehicle_year = st.number_input("Vehicle Year", min_value=1900, max_value=2025, value=2015)
            route_type = st.selectbox("Route Type", ["Interstate (State)", "US (State)", "Maryland (State) Route", "County Route", "Other"])
        
        # Additional inputs needed for feature engineering
        st.subheader("Additional Information")
        col3, col4 = st.columns(2)
        with col3:
            crash_hour = st.slider("Crash Hour (0-23)", 0, 23, 12)
            crash_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        with col4:
            traffic_control = st.selectbox("Traffic Control", ["No Controls", "Traffic Control Signal", "Stop Sign", "Yield Sign", "Other"])
            vehicle_damage = st.selectbox("Vehicle Damage Extent", ["No Damage", "Superficial", "Functional", "Disabling", "Vehicle Not at Scene"])
            vehicle_body = st.selectbox("Vehicle Body Type", ["Passenger Car", "Sport Utility Vehicle", "Pickup", "Van - Passenger (<9 Seats)", "Other"])
            vehicle_movement = st.selectbox("Vehicle Movement", ["Moving Constant Speed", "Slowing or Stopping", "Stopped in Traffic", "Turning Left", "Turning Right", "Other"])
        
        driver_substance = st.selectbox("Driver Substance Abuse", ["Not Suspect of Alcohol Use, Not Suspect of Drug Use", "Suspect of Alcohol Use", "Unknown"])
        
        # Predict button
        if st.button("Predict Injury Severity", type="primary"):
            try:
                # Create a function to prepare features
                def prepare_features(user_inputs):
                    """Prepare features matching the training pipeline"""
                    # Map day of week to number
                    day_map = {
                        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                        "Friday": 4, "Saturday": 5, "Sunday": 6
                    }
                    
                    # Create base DataFrame
                    data = {
                        'Weather': [user_inputs['weather']],
                        'Surface Condition': [user_inputs['surface']],
                        'Light': [user_inputs['light']],
                        'Traffic Control': [user_inputs['traffic_control']],
                        'Collision Type': [user_inputs['collision_type']],
                        'Driver At Fault': [user_inputs['driver_at_fault']],
                        'Driver Substance Abuse': [user_inputs['driver_substance']],
                        'Vehicle Damage Extent': [user_inputs['vehicle_damage']],
                        'Vehicle Body Type': [user_inputs['vehicle_body']],
                        'Vehicle Movement': [user_inputs['vehicle_movement']],
                        'Speed Limit': [user_inputs['speed_limit']],
                        'Route Type': [user_inputs['route_type']],
                        'Vehicle Year': [user_inputs['vehicle_year']],
                        'Crash Hour': [user_inputs['crash_hour']],
                        'Crash DayOfWeek': [day_map[user_inputs['crash_day']]]
                    }
                    
                    df_input = pd.DataFrame(data)
                    
                    # Feature engineering: Vehicle Age
                    current_year = 2025
                    df_input['Vehicle Age'] = current_year - df_input['Vehicle Year'].replace(0, np.nan)
                    df_input['Vehicle Age'] = df_input['Vehicle Age'].fillna(df_input['Vehicle Age'].median())
                    
                    # Feature engineering: Has_Substance_Abuse
                    df_input['Has_Substance_Abuse'] = df_input['Driver Substance Abuse'].apply(
                        lambda x: 1 if pd.notna(x) and 'Suspect' in str(x) else 0
                    )
                    
                    # Remove Vehicle Year (replaced by Vehicle Age)
                    df_input = df_input.drop(columns=['Vehicle Year'])
                    
                    # Select features (matching training)
                    selected_features = [
                        'Weather', 'Surface Condition', 'Light', 'Traffic Control',
                        'Collision Type', 'Driver At Fault', 'Driver Substance Abuse',
                        'Vehicle Damage Extent', 'Vehicle Body Type', 'Vehicle Movement',
                        'Speed Limit', 'Route Type', 'Crash Hour', 'Crash DayOfWeek',
                        'Vehicle Age', 'Has_Substance_Abuse'
                    ]
                    
                    X = df_input[selected_features].copy()
                    
                    # Fill missing values
                    for col in selected_features:
                        if col in X.columns:
                            if X[col].dtype == 'object':
                                X[col] = X[col].fillna('Unknown')
                            else:
                                X[col] = X[col].fillna(X[col].median())
                    
                    # Separate categorical and numerical
                    categorical_features = [f for f in selected_features if X[f].dtype == 'object']
                    numerical_features = [f for f in selected_features if X[f].dtype != 'object']
                    
                    # One-hot encode categorical variables
                    X_encoded = pd.get_dummies(X[categorical_features], prefix=categorical_features, drop_first=True)
                    
                    # Combine with numerical features
                    X_final = pd.concat([X[numerical_features], X_encoded], axis=1)
                    
                    return X_final
                
                # Prepare user inputs
                user_inputs = {
                    'weather': weather,
                    'surface': surface,
                    'light': light,
                    'traffic_control': traffic_control,
                    'collision_type': collision_type,
                    'driver_at_fault': driver_at_fault,
                    'driver_substance': driver_substance,
                    'vehicle_damage': vehicle_damage,
                    'vehicle_body': vehicle_body,
                    'vehicle_movement': vehicle_movement,
                    'speed_limit': speed_limit,
                    'route_type': route_type,
                    'vehicle_year': vehicle_year,
                    'crash_hour': crash_hour,
                    'crash_day': crash_day
                }
                
                # Prepare features
                X_input = prepare_features(user_inputs)
                
                # Try to align features with training set
                # Get model's expected features from the model itself
                try:
                    # Try to get feature names from preprocessing info
                    preprocessing_info = joblib.load('preprocessing_info.pkl')
                    expected_features = preprocessing_info.get('feature_names', [])
                    
                    if len(expected_features) > 0:
                        # Align features with training set
                        # Add missing columns with 0 (these are one-hot encoded features not present in this input)
                        for feat in expected_features:
                            if feat not in X_input.columns:
                                X_input[feat] = 0
                        
                        # Keep only expected features and reorder
                        X_input = X_input.reindex(columns=expected_features, fill_value=0)
                    else:
                        st.warning("⚠️ Feature names not found in preprocessing info.")
                        
                except Exception as e:
                    # If preprocessing info not available, try to infer from model
                    st.warning(f"⚠️ Could not load preprocessing info: {str(e)}. Using current features.")
                    # The model will handle feature mismatch if it's tree-based (RF, GB)
                
                # Make prediction
                if st.session_state.model is not None:
                    # Check if model needs scaling (Logistic Regression)
                    if st.session_state.scaler is not None:
                        X_input_scaled = st.session_state.scaler.transform(X_input)
                        prediction = st.session_state.model.predict(X_input_scaled)
                    else:
                        prediction = st.session_state.model.predict(X_input)
                    
                    # Get prediction probabilities if available
                    try:
                        if st.session_state.scaler is not None:
                            proba = st.session_state.model.predict_proba(X_input_scaled)
                        else:
                            proba = st.session_state.model.predict_proba(X_input)
                        
                        # Display results
                        st.success(f"## Predicted Injury Severity: **{prediction[0]}**")
                        
                        st.subheader("Prediction Probabilities:")
                        proba_df = pd.DataFrame({
                            'Injury Severity': st.session_state.model.classes_,
                            'Probability': proba[0]
                        }).sort_values('Probability', ascending=False)
                        st.dataframe(proba_df, use_container_width=True)
                        
                        # Visualize probabilities
                        fig, ax = plt.subplots(figsize=(10, 6))
                        proba_df.plot(x='Injury Severity', y='Probability', kind='barh', ax=ax, legend=False)
                        ax.set_title('Prediction Probabilities')
                        ax.set_xlabel('Probability')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    except:
                        st.success(f"## Predicted Injury Severity: **{prediction[0]}**")
                else:
                    st.error("❌ Model not loaded. Please load the model first.")
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("💡 Make sure all inputs are filled and the model is trained in the notebook.")

# Standard Python script entry point.
if __name__ == "__main__":
    pass
