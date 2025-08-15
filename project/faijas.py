import streamlit as st
import joblib
import numpy as np

# Load model and encoders
Xgbc = joblib.load('df_X (2).pkl')
scaler = joblib.load("scaler.pkl")
le_marital = joblib.load("le1.pkl")
le_job = joblib.load("le2.pkl")

# Static dictionaries for mapping
education_map = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
binary_map = {'yes': 1, 'no': 0}
poutcome_map = {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3}
contact_map = {'unknown': 0, 'cellular': 1, 'telephone': 2}

st.title("Bank Term Deposit Subscription Predictor")
st.header("Enter Client Information")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100)
job = st.selectbox("Job", le_job.classes_)
marital = st.selectbox("Marital Status", le_marital.classes_)
education = st.selectbox("Education", list(education_map.keys()))
default = st.selectbox("Credit in Default?", list(binary_map.keys()))
balance = st.number_input("Account Balance")
housing = st.selectbox("Has Housing Loan?", list(binary_map.keys()))
loan = st.selectbox("Has Personal Loan?", list(binary_map.keys()))
contact = st.selectbox("Contact Type", list(contact_map.keys()))
campaign = st.number_input("Number of Contacts During Campaign", min_value=0)
previous = st.number_input("Number of Contacts Before Campaign", min_value=0)
poutcome = st.selectbox("Previous Campaign Outcome", list(poutcome_map.keys()))

# Encoding
job_enc = le_job.transform([job])[0]
marital_enc = le_marital.transform([marital])[0]
education_enc = education_map[education]
default_enc = binary_map[default]
housing_enc = binary_map[housing]
loan_enc = binary_map[loan]
contact_enc = contact_map[contact]
poutcome_enc = poutcome_map[poutcome]

# Final feature list
features = np.array([[age, job_enc, marital_enc, education_enc, default_enc,
                      balance, housing_enc, loan_enc, contact_enc, campaign,
                      previous, poutcome_enc]])

# Prediction
if st.button("Predict"):
    features_scaled = scaler.transform(features)
    prediction = Xgbc.predict(features_scaled)[0]
    if prediction == 1:
        st.success("✅ The client is likely to **subscribe** to a term deposit.")
    else:
        st.warning("❌ The client is **not likely** to subscribe.")
