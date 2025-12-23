import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load trained artifacts
# -------------------------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="🧑‍💼",
    layout="centered"
)

st.title("🧑‍💼 Employee Retention Prediction")
st.write("Predict employee attrition risk for proactive HR decision-making")

st.divider()

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.subheader("📋 Employee Details")

city = st.number_input("City Code", 0, 200, 10)
city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.7)

gender = st.selectbox("Gender", ["Female", "Male", "Other"])
relevent_experience = st.selectbox(
    "Relevant Experience",
    ["No relevent experience", "Has relevent experience"]
)
enrolled_university = st.selectbox(
    "Enrolled University",
    ["no_enrollment", "Part time course", "Full time course"]
)
education_level = st.selectbox(
    "Education Level",
    ["Primary School", "High School", "Graduate", "Masters", "Phd"]
)
major_discipline = st.selectbox(
    "Major Discipline",
    ["STEM", "Business Degree", "Arts", "Humanities", "Other"]
)

experience = st.slider("Years of Experience", 0, 30, 5)

company_size = st.selectbox(
    "Company Size",
    [
        "<10", "10-49", "50-99", "100-500",
        "500-999", "1000-4999", "5000-9999", "10000+"
    ]
)

company_type = st.selectbox(
    "Company Type",
    ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Other"]
)

last_new_job = st.selectbox(
    "Years Since Last Job Change",
    ["Never", "1", "2", "3", "4", ">4"]
)

training_hours = st.slider("Training Hours", 0, 300, 40)

# -------------------------------------------------
# Label Encoding (same as notebook)
# -------------------------------------------------
gender_map = {"Female": 0, "Male": 1, "Other": 2}
relevant_exp_map = {
    "No relevent experience": 0,
    "Has relevent experience": 1
}
enrolled_map = {
    "no_enrollment": 0,
    "Part time course": 1,
    "Full time course": 2
}
education_map = {
    "Primary School": 0,
    "High School": 1,
    "Graduate": 2,
    "Masters": 3,
    "Phd": 4
}
discipline_map = {
    "STEM": 0,
    "Business Degree": 1,
    "Arts": 2,
    "Humanities": 3,
    "Other": 4
}
company_size_map = {
    "<10": 0, "10-49": 1, "50-99": 2, "100-500": 3,
    "500-999": 4, "1000-4999": 5, "5000-9999": 6, "10000+": 7
}
company_type_map = {
    "Pvt Ltd": 0, "Funded Startup": 1,
    "Public Sector": 2, "NGO": 3, "Other": 4
}
last_job_map = {
    "Never": 0, "1": 1, "2": 2, "3": 3, "4": 4, ">4": 5
}

# -------------------------------------------------
# Create input DataFrame
# -------------------------------------------------
input_df = pd.DataFrame([{
    "city": city,
    "city_development_index": city_development_index,
    "gender": gender_map[gender],
    "relevent_experience": relevant_exp_map[relevent_experience],
    "enrolled_university": enrolled_map[enrolled_university],
    "education_level": education_map[education_level],
    "major_discipline": discipline_map[major_discipline],
    "experience": experience,
    "company_size": company_size_map[company_size],
    "company_type": company_type_map[company_type],
    "last_new_job": last_job_map[last_new_job],
    "training_hours": training_hours
}])

# -------------------------------------------------
# Add columns expected by scaler & model
# -------------------------------------------------
input_df["enrollee_id"] = 0   # model expects this
input_df["target"] = 0        # scaler expects this

# -------------------------------------------------
# Reorder columns exactly as scaler was trained
# -------------------------------------------------
input_df = input_df[scaler.feature_names_in_]

# -------------------------------------------------
# Apply scaler
# -------------------------------------------------
input_df_scaled = pd.DataFrame(
    scaler.transform(input_df),
    columns=scaler.feature_names_in_
)

# -------------------------------------------------
# Remove ONLY target (model does not need it)
# -------------------------------------------------
input_df_scaled.drop(columns=["target"], inplace=True)

# -------------------------------------------------
# Align exactly with model features (NOW enrollee_id exists)
# -------------------------------------------------
input_df_scaled = input_df_scaled[model.feature_name_]

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.divider()

if st.button("🔍 Predict Attrition Risk"):
    probability = model.predict_proba(input_df_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    if probability < 0.30:
        st.success(f"🟢 Low Attrition Risk\n\nProbability: {probability:.2f}")
    elif probability < 0.70:
        st.warning(f"🟡 Medium Attrition Risk\n\nProbability: {probability:.2f}")
    else:
        st.error(f"🔴 High Attrition Risk\n\nProbability: {probability:.2f}")

st.divider()
st.caption("Capstone Project | Employee Retention Prediction | Streamlit App")
