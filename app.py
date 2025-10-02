import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Passenger Satisfaction Prediction", layout="centered")
st.title("✈️ Passenger Satisfaction Prediction")

# ---------------------------
# Load model and artifacts
# ---------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ---------------------------
# EDA Visualizations
# ---------------------------
st.subheader("📊 Exploratory Data Analysis")

eda_images = {
    "Distribution Plots": "eda_distribution_plots.png",
    "Categorical Satisfaction": "categorical_satisfaction.png",
    "Correlation Heatmap": "correlation_heatmap.png"
}

for title, path in eda_images.items():
    if os.path.exists(path):
        st.markdown(f"**{title}**")
        st.image(path, use_container_width=True)
    else:
        st.warning(f"EDA image not found: `{path}`. Please add it to the app folder.")

# ---------------------------
# EDA Summary
# ---------------------------
st.subheader("📝 EDA Summary")

st.markdown("""
- **Age Distribution:** The majority of passengers are between 25–50 years old, with some younger and older outliers.
- **Flight Distance:** Most flights are short to medium range (under 2000 km), but a few long-haul flights are present.
- **Arrival Delay:** There is a long tail in delays, with most flights on time or slightly delayed, but some have delays over 400 minutes.

- **Satisfaction by Gender:** There is no strong gender bias in satisfaction levels.
- **Satisfaction by Customer Type:** Loyal customers are significantly more likely to be satisfied compared to disloyal ones.
- **Satisfaction by Type of Travel:** Business travelers tend to report higher satisfaction compared to personal travelers.

- **Correlation Heatmap:**
  - Features like **Inflight entertainment**, **Online boarding**, and **Seat comfort** show strong positive correlation with satisfaction.
  - **Arrival delay** and **departure delay** have weak or no correlation with satisfaction, suggesting that service quality matters more.

📌 Overall, service features (e.g., cleanliness, entertainment, boarding) influence satisfaction more than delays or demographics.
""")

# ---------------------------
# Input Form
# ---------------------------
st.subheader("📟 Enter Passenger Details")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
        travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
        flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        age = st.slider("Age", 7, 85, 30)

    with col2:
        flight_distance = st.slider("Flight Distance", 30, 5000, 500)
        dep_delay = st.slider("Departure Delay (min)", 0, 500, 5)
        arr_delay = st.slider("Arrival Delay (min)", 0, 500, 10)

    st.markdown("**Service Ratings (0–5):**")
    wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 4)
    online_boarding = st.slider("Online Boarding", 0, 5, 4)
    entertainment = st.slider("Inflight Entertainment", 0, 5, 3)

    # Additional ratings
    gate_location = st.slider("Gate Location", 0, 5, 3)
    food = st.slider("Food and Drink", 0, 5, 3)
    seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
    onboard_service = st.slider("On-board Service", 0, 5, 3)
    legroom = st.slider("Leg Room Service", 0, 5, 3)
    baggage = st.slider("Baggage Handling", 0, 5, 3)
    checkin = st.slider("Check-in Service", 0, 5, 3)
    inflight_service = st.slider("Inflight Service", 0, 5, 3)
    booking = st.slider("Ease of Online Booking", 0, 5, 3)
    dep_arr_convenient = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)

    submitted = st.form_submit_button("Predict Satisfaction")

# ---------------------------
# Inference Logic
# ---------------------------
if submitted:
    # Encode inputs (same as before)
    input_data = {
        'Gender': label_encoders['Gender'].transform([gender])[0],
        'Customer Type': label_encoders['Customer Type'].transform([customer_type])[0],
        'Type of Travel': label_encoders['Type of Travel'].transform([travel_type])[0],
        'Age': age,
        'Flight Distance': flight_distance,
        'Departure Delay in Minutes': dep_delay,
        'Arrival Delay in Minutes': arr_delay,
        'Inflight wifi service': wifi,
        'Cleanliness': cleanliness,
        'Online boarding': online_boarding,
        'Inflight entertainment': entertainment,
        'Gate location': gate_location,
        'Food and drink': food,
        'Seat comfort': seat_comfort,
        'On-board service': onboard_service,
        'Leg room service': legroom,
        'Baggage handling': baggage,
        'Checkin service': checkin,
        'Inflight service': inflight_service,
        'Ease of Online booking': booking,
        'Departure/Arrival time convenient': dep_arr_convenient
    }

    class_onehot = {'Class_Eco': 0, 'Class_Eco Plus': 0}
    if flight_class == 'Eco':
        class_onehot['Class_Eco'] = 1
    elif flight_class == 'Eco Plus':
        class_onehot['Class_Eco Plus'] = 1

    input_data.update(class_onehot)

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Get probability of class 1 ("Satisfied")
    proba = model.predict_proba(input_df)[0][1]

    # Add threshold slider
    threshold = st.slider("Select classification threshold", 0.0, 1.0, 0.5, 0.01)

    # Determine predicted class based on threshold
    prediction = 1 if proba >= threshold else 0
    result = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"

    st.success(f"Prediction: **{result}** with confidence {proba:.2f} ✈️")

    if proba < threshold:
        st.info(f"The model probability is below the selected threshold ({threshold}). You may try adjusting inputs or lowering threshold.")
