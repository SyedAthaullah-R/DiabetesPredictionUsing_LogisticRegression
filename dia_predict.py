

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV data into a Pandas DataFrame
dia_data = pd.read_csv(r"F:\Diabetes prediction\diabetes_risk_prediction_dataset (1).csv")

# Split the data into features (X) and target (Y)
X = dia_data.drop(columns='class', axis=1)
Y = dia_data['class']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Streamlit app with background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.example.com/background.jpg");
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background: url("https://www.example.com/sidebar-background.jpg");
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; font-size:35px;'>Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.text("            by Syed Athaullah")

# Input validation
age = st.number_input("Age", 5, 99)
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
polyuria = st.selectbox("Polyuria (Frequent urination)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
polydipsia = st.selectbox("Polydipsia (Excessive thirst)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
sudden_weight_loss = st.selectbox("Sudden weight loss", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
weakness = st.selectbox("Weakness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
polyphagia = st.selectbox("Polyphagia (Excessive hunger)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
genital_thrush = st.selectbox("Genital thrush", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
visual_blurring = st.selectbox("Visual blurring", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
itching = st.selectbox("Itching", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
irritability = st.selectbox("Irritability", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
delayed_healing = st.selectbox("Delayed healing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
partial_paresis = st.selectbox("Partial paresis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
muscle_stiffness = st.selectbox("Muscle stiffness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alopecia = st.selectbox("Alopecia (Hair loss)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

input_data = (age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, genital_thrush,
              visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity)

if st.button("Predict Diabetes Risk"):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 0:
        st.success('The person does not have diabetes')
    else:
        st.success('The person has diabetes')

