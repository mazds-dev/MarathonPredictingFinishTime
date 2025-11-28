import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------
# Load trained model and feature column names
# ---------------------------------------------
# These files were saved from the Jupyter Notebook after training the model.
model = joblib.load("marathon_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------------------------------------------
# Streamlit page configuration
# ---------------------------------------------
st.set_page_config(
    page_title="Marathon Finish Time Predictor",
    layout="centered",
    page_icon="🏃‍➡️"
)

# ---------------------------------------------
# App Title and Description
# ---------------------------------------------
st.title("Marathon Finish Time Predictor 🏃‍♂️‍➡️")
st.caption("Predict your marathon finish time using a machine learning model trained on the 2023 Marathon Results dataset.")

# ---------------------------------------------
# Sidebar: About the App
# ---------------------------------------------
st.sidebar.header("About this App")
st.sidebar.write(
    """
    This interactive app uses a **Linear Regression model** trained on over 420,000 marathon results.

    It predicts marathon finish times using:
    - **Age**
    - **Gender**
    - **Race**

    The model was trained in Python using:
    - Pandas  
    - Scikit-learn  
    - Seaborn  
    - Matplotlib  

    View the full analysis and notebook on GitHub.
    """
)

# ---------------------------------------------
# User Input Section
# ---------------------------------------------
st.subheader("🔢 Enter Your Information")

# Columns for cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=16, max_value=90, value=30)

with col2:
    gender_display = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_display == "Male" else 0

# ---------------------------------------------
# Race Selection
# ---------------------------------------------
st.write("### 🏁 Select Your Race")

# List of popular races (clean and user-friendly)
popular_races = [
    "Boston Marathon",
    "Chicago Marathon",
    "New York City Marathon",
    "Los Angeles Marathon",
    "Houston Marathon",
    "Philadelphia Marathon",
    "Twin Cities Marathon",
    "Marine Corps Marathon",
]

race_input = st.selectbox(
    "Choose a race or select 'Other (type manually)'",
    popular_races + ["Other (type manually)"]
)

# If user selects "Other", show a text input
if race_input == "Other (type manually)":
    race_input = st.text_input("Enter race name exactly as in the dataset:")

# Warning if field is empty
if race_input == "":
    st.warning("Please type a race name or choose one from the list.")

# ---------------------------------------------
# Prepare Input for the Model
# ---------------------------------------------
input_data = {
    "Age": age,
    "Gender": gender,
    "Race": race_input
}

input_df = pd.DataFrame([input_data])

# One-hot encode the race column
input_encoded = pd.get_dummies(input_df, columns=["Race"])

# Add missing columns so the model receives the expected feature set
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure correct column order
input_encoded = input_encoded[model_columns]

# ---------------------------------------------
# Prediction Button
# ---------------------------------------------
if st.button("Predict Finish Time"):
    
    # Make prediction (in seconds)
    prediction_seconds = int(model.predict(input_encoded)[0])

    # Convert to hours/minutes/seconds
    hours = prediction_seconds // 3600
    minutes = (prediction_seconds % 3600) // 60
    seconds = prediction_seconds % 60

    # -----------------------------------------
    # Display Results
    # -----------------------------------------
    st.success(f"🥇 **Estimated Finish Time: {hours}h {minutes}m {seconds}s**")

    # Show pace estimates
    pace_per_km = prediction_seconds / 42.195
    pace_per_mile = prediction_seconds / 26.2

    st.write(f"**Pace per km:** {int(pace_per_km//60)}:{int(pace_per_km%60):02d} min/km")
    st.write(f"**Pace per mile:** {int(pace_per_mile//60)}:{int(pace_per_mile%60):02d} min/mile")

    # Prediction summary
    st.subheader("📋 Prediction Summary")
    st.write(
        f"""
        - **Age:** {age}  
        - **Gender:** {gender_display}  
        - **Race:** {race_input}  
        - **Predicted Finish Time:** {hours}h {minutes}m 
        """
    )

    st.info("⚠️ Note: Actual performance may vary depending on training, pacing strategy, weather, and terrain.")

# ---------------------------------------------
# Footer: Link to GitHub
# ---------------------------------------------
st.markdown("---")
st.markdown("**🔎 Want to see the full project, code, analysis, and graphs?**  👉 [View the Jupyter Notebook on GitHub](https://github.com/mazds-dev/MarathonPredictingFinishTime)")

# ---------------------------------------------
# My details
# ---------------------------------------------

with st.expander("👤 About the Author"):
    st.write("""
    **Marvin Adorian Zanchi Santos**  
    BSc in Software Development | SEM1 - Year 4 - 2025  
    South East Technological University (SETU), Carlow Campus  
    Module: Data Science & Machine Learning 1  
    Lecturer: Ben O’Shaughnessy  
    """)


