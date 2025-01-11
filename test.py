import pandas as pd
import joblib

# Load the trained model
model = joblib.load('decision_tree_model.pkl')  # Update the path to your model file


# Function to get input data from the user
def get_user_input():
    # Ask the user for inputs, one by one
    age = int(input("Enter age: "))
    education = int(input("Enter education level (e.g., 0 for none, 1 for high school, 2 for college, etc.): "))
    sex = int(input("Enter sex (1 for male, 0 for female): "))
    is_smoking = int(input("Are you smoking? (1 for yes, 0 for no): "))
    cigsPerDay = int(input("How many cigarettes per day? "))
    BPMeds = int(input("Do you take blood pressure medications? (1 for yes, 0 for no): "))
    prevalentStroke = int(input("Do you have a history of stroke? (1 for yes, 0 for no): "))
    prevalentHyp = int(input("Do you have hypertension? (1 for yes, 0 for no): "))
    diabetes = int(input("Do you have diabetes? (1 for yes, 0 for no): "))
    totChol = int(input("Enter total cholesterol level: "))
    sysBP = int(input("Enter systolic blood pressure (sysBP): "))
    diaBP = int(input("Enter diastolic blood pressure (diaBP): "))
    BMI = float(input("Enter BMI: "))
    heartRate = int(input("Enter heart rate: "))
    glucose = int(input("Enter glucose level: "))

    # Return the inputs as a pandas DataFrame
    input_data = pd.DataFrame([{
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

    return input_data


# Function to make predictions
def make_predictions(model):
    # Get input data from the user
    input_df = get_user_input()

    # Make predictions using the trained model
    predictions = model.predict(input_df)

    # Display the prediction with a message
    if predictions[0] == 1:
        print("Prediction: You may have a heart disease.")
    else:
        print("Prediction: You are not at risk for heart disease.")


# Loop to give users the option for multiple predictions
while True:
    make_predictions(model)

    # Ask if the user wants to make another prediction
    another = input("\nDo you want to make another prediction? (yes/no): ").lower()
    if another != 'yes':
        print("Thank you for using the prediction tool!")
        break
