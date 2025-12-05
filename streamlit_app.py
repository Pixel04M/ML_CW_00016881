"""
Streamlit Multi-Page Application for Crash Reporting Analysis

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
    ["🏠 Home", "📊 Data Exploration", "🔧 Preprocessing", "🤖 Model Training", "📈 Model Evaluation", "🔮 Prediction"]
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
    2. Use **Preprocessing** to clean and prepare the data  
    3. Train models under **Model Training**  
    4. Compare performance in **Model Evaluation**  
    5. Use **Prediction** to make a new prediction  
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

# ------------------ PREPROCESSING PAGE --------------------
elif page == "🔧 Preprocessing":
    st.title("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load the dataset from the Home page first.")
    else:
        df = st.session_state.data.copy()
        
        # This page only explains preprocessing; full pipeline is in the notebook.
        st.info("💡 This page shows an overview of the preprocessing steps used in the notebook.")
        
        st.subheader("Preprocessing Steps Applied:")
        st.markdown("""
        1. Filled missing values  
        2. Standardized numerical features  
        3. Removed duplicates and invalid records  
        4. Created engineered features (vehicle age, crash hour, etc.)  
        5. Applied one-hot encoding  
        6. Split data into train/validation/test  
        """)

# ------------------ MODEL TRAINING PAGE -------------------
elif page == "🤖 Model Training":
    st.title(" Model Training")
    
    # Explain training process
    st.info(" Models were trained in the notebook using GridSearchCV.")
    
    st.subheader("Models Trained:")
    st.markdown("""
    - Random Forest  
    - Gradient Boosting  
    - Logistic Regression  
    (All tuned using GridSearchCV)
    """)
    
    # Try loading model (only once, not on every rerun)
    if st.session_state.model is None:
        try:
            model = joblib.load('best_model.pkl')
            st.session_state.model = model
            st.success("✅ Best model loaded!")
            
            # Load scaler if exists
            try:
                scaler = joblib.load('scaler.pkl')
                st.session_state.scaler = scaler
            except:
                pass
        except Exception as e:
            st.warning(f"⚠️ Model file not found: {str(e)}. Train models in the notebook first.")
    else:
        st.success("✅ Model already loaded!")

# ------------------ MODEL EVALUATION PAGE -----------------
elif page == "📈 Model Evaluation":
    st.title(" Model Evaluation")
    
    st.subheader("Evaluation Metrics Used:")
    st.markdown("""
    - Accuracy  
    - Precision  
    - Recall  
    - F1-Score  
    - Confusion Matrix  
    - Classification Report  
    """)
    
    st.info("💡 Full evaluation results can be found in the Jupyter notebook.")

# ---------------------- PREDICTION PAGE --------------------
elif page == "🔮 Prediction":
    st.title("🔮 Make Predictions")
    
    # Ensure model is loaded
    if st.session_state.model is None:
        st.warning("⚠️ Load the model from the Model Training page first.")
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
        
        # Predict button
        if st.button("Predict Injury Severity", type="primary"):
            st.info(" Inputs must match the original training pipeline exactly.")
            st.success("Prediction requires full feature engineering (see notebook).")

# Standard Python script entry point.
if __name__ == "__main__":
    pass
