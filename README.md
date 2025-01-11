# Heart Disease Prediction Model

## Overview

This project aims to predict the likelihood of heart disease based on various health-related features. The model utilizes a dataset of patient information, including age, cholesterol levels, blood pressure, and lifestyle factors such as smoking habits. The heart disease prediction is binary, indicating whether a person is at risk for a heart disease (1) or not (0).

## Dataset

The dataset used for training and testing the model is based on health data and contains the following columns:

- **age**: Age of the patient
- **education**: Level of education (encoded as an integer)
- **sex**: Gender of the patient (encoded: 0 for male, 1 for female)
- **is_smoking**: Whether the patient smokes (encoded: 0 for no, 1 for yes)
- **cigsPerDay**: Average number of cigarettes smoked per day
- **BPMeds**: Whether the patient uses blood pressure medication (0 for no, 1 for yes)
- **prevalentStroke**: Whether the patient has had a stroke (0 for no, 1 for yes)
- **prevalentHyp**: Whether the patient has hypertension (0 for no, 1 for yes)
- **diabetes**: Whether the patient has diabetes (0 for no, 1 for yes)
- **totChol**: Total cholesterol level
- **sysBP**: Systolic blood pressure
- **diaBP**: Diastolic blood pressure
- **BMI**: Body Mass Index
- **heartRate**: Heart rate
- **glucose**: Glucose level
- **TenYearCHD**: Target variable indicating heart disease risk (0 for no, 1 for yes)

## Setup

### Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- pickle

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
Data Preprocessing:
The dataset is first loaded, and unnecessary columns (like the 'id' column) are removed. Missing values are imputed using the most frequent strategy or the mean for numerical columns.

Model Training:
A decision tree classifier is used for prediction. The data is split into training and testing sets, and SMOTE is applied to address class imbalance.

Model Tuning:
GridSearchCV is used to tune hyperparameters for the decision tree classifier, specifically optimizing for accuracy.

Prediction:
Once the model is trained, you can predict the likelihood of heart disease by providing an input sample using the same features as in the dataset.

python
Copy code
# Example code for making a prediction
input_df = pd.DataFrame([{
    'age': 45,
    'education': 2,
    'sex': 1,
    'is_smoking': 1,
    'cigsPerDay': 15,
    'BPMeds': 1,
    'prevalentStroke': 0,
    'prevalentHyp': 0,
    'diabetes': 0,
    'totChol': 240,
    'sysBP': 130,
    'diaBP': 85,
    'BMI': 26.0,
    'heartRate': 72,
    'glucose': 90
}])

prediction = model.predict(input_df)
print(f"Heart Disease Risk Prediction: {prediction[0]}")
Saving and Loading the Model:
The trained model is saved using pickle, and you can load it for future use.

python
Copy code
# Save the model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)
Results
After applying SMOTE to balance the dataset, the Decision Tree model achieved a best accuracy of 82.6% using GridSearchCV.

Conclusion
This model successfully predicts the likelihood of heart disease based on key health factors. It can be used as part of a larger health monitoring system to assess the risk for individuals and offer personalized health recommendations.

License
This project is licensed under the MIT License.

Acknowledgements
The dataset used in this project is publicly available and was used for educational purposes.
Special thanks to the authors of the imbalanced-learn library for the SMOTE implementation.
