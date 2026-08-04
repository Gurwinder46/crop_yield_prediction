import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Page setup
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Files
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.pkl"
DATA_PATH = BASE_DIR / "data" / "crop_yield.csv"

# Load model and dataset
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Sidebar
st.sidebar.title("Crop Yield Prediction")

page = st.sidebar.radio(
    "Menu",
    ["Home", "Prediction", "Data Analysis", "About"]
)

# ================= HOME =================

if page == "Home":

    st.title("🌾 Crop Yield Prediction")

    st.write(
        "A Machine Learning application for predicting crop yield "
        "using agricultural and environmental parameters."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Random Forest")
    col2.metric("R² Score", "94.68%")
    col3.metric("Dataset", "1000 Records")

    st.divider()

    st.subheader("Project Features")

    st.write("""
    • Crop yield prediction  
    • Agricultural data analysis  
    • Machine Learning model  
    • Interactive web interface
    """)

# ================= PREDICTION =================

elif page == "Prediction":

    st.title("🌾 Crop Yield Prediction")

    st.write("Enter the following agricultural details.")

    col1, col2 = st.columns(2)

    with col1:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=150.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            value=28.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    with col2:
        fertilizer = st.number_input(
            "Fertilizer (kg/hectare)",
            min_value=0.0,
            value=100.0
        )

        pesticide = st.number_input(
            "Pesticide (kg/hectare)",
            min_value=0.0,
            value=15.0
        )

        crop = st.selectbox(
            "Crop",
            ["Maize", "Rice", "Wheat"]
        )

    st.divider()

    if st.button("Predict Yield"):

        crop_mapping = {
            "Maize": 0,
            "Rice": 1,
            "Wheat": 2
        }

        input_data = pd.DataFrame({
            "Rainfall": [rainfall],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Fertilizer": [fertilizer],
            "Pesticide": [pesticide],
            "Crop": [crop_mapping[crop]]
        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Yield: {prediction:.2f} tons/hectare"
        )

# ================= DATA ANALYSIS =================

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("Crop Distribution")

    fig, ax = plt.subplots()

    df["Crop"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Crop")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)

    plt.close(fig)

# ================= ABOUT =================

elif page == "About":

    st.title("About the Project")

    st.subheader("Objective")

    st.write(
        "To develop a Machine Learning model that predicts crop yield "
        "using agricultural and environmental factors."
    )

    st.subheader("Algorithm")

    st.write("Random Forest Regressor")

    st.subheader("Technologies")

    st.write(
        "Python, Pandas, NumPy, Matplotlib, Scikit-learn, "
        "Joblib and Streamlit."
    )

    st.subheader("Input Parameters")

    st.write(
        "Rainfall, Temperature, Humidity, Fertilizer, "
        "Pesticide and Crop Type."
    )

    st.subheader("Output")

    st.write("Predicted crop yield in tons per hectare.")