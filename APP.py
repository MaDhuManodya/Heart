import pandas as pd
import joblib
import streamlit as st

# Load the trained model
model = joblib.load('decision_tree_model.pkl')  # Update the path to your model file

# Function to make predictions
def make_predictions(input_data):
    # Make predictions using the trained model
    predictions = model.predict(input_data)

    # Display the prediction with a message
    if predictions[0] == 1:
        return "You may have a heart disease. ❤️"
    else:
        return "You are not at risk for heart disease. 😊"

# Streamlit UI
st.title("Heart Disease Prediction 🫀")

# Add an image (you can use your own image path or a URL)
st.image("image.png", caption="Heart Disease Awareness", use_container_width=True)

# Layout: Create two columns
col1, col2 = st.columns(2)

# Collect input from the user
with col1:
    age = st.number_input("👤 Enter age:", min_value=0, max_value=120)
    education = st.selectbox(
        "🎓 Enter education level:",
        options=[0, 1, 2],
        index=0,
        format_func=lambda x: "None" if x == 0 else "High School" if x == 1 else "College")
    sex = st.selectbox("🚹 Enter sex:", options=[1, 0], index=0, format_func=lambda x: "Male" if x == 1 else "Female")
    is_smoking = st.selectbox("🚬 Are you smoking?", options=[1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    cigsPerDay = st.number_input("🚭 How many cigarettes per day?", min_value=0, max_value=100, value=0)
    BPMeds = st.selectbox("💊 Do you take blood pressure medications?", options=[1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    prevalentStroke = st.selectbox("🧠 Do you have a history of stroke?", options=[1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    prevalentHyp = st.selectbox("💔 Do you have hypertension?", options=[1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("🩸 Do you have diabetes?", options=[1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    totChol = st.number_input("🧪 Enter total cholesterol level:", min_value=100, max_value=500, value=200)
    sysBP = st.number_input("💓 Enter systolic blood pressure (sysBP):", min_value=80, max_value=200, value=120)
    diaBP = st.number_input("🩺 Enter diastolic blood pressure (diaBP):", min_value=40, max_value=120, value=80)

# Collect remaining inputs
BMI = st.number_input("🏋️‍♂️ Enter BMI:", min_value=10.0, max_value=50.0, value=25.0)
heartRate = st.number_input("❤️ Enter heart rate:", min_value=40, max_value=200, value=72)
glucose = st.number_input("🩸 Enter glucose level:", min_value=50, max_value=300, value=90)

# Create the input DataFrame for prediction
input_df = pd.DataFrame([{
    'age': age,
    'education': education,
    'sex': sex,
    'is_smoking': is_smoking,
    'cigsPerDay': cigsPerDay,
    'BPMeds': BPMeds,
    'prevalentStroke': prevalentStroke,
    'prevalentHyp': prevalentHyp,
    'diabetes': diabetes,
    'totChol': totChol,
    'sysBP': sysBP,
    'diaBP': diaBP,
    'BMI': BMI,
    'heartRate': heartRate,
    'glucose': glucose
}], columns=['age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
             'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
             'diaBP', 'BMI', 'heartRate', 'glucose'])

# Button to make the prediction
if st.button("🔮 Predict"):
    prediction_message = make_predictions(input_df)
    st.success(f"Prediction: {prediction_message}")
