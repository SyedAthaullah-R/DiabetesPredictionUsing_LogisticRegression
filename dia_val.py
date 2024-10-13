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

# Streamlit app
st.markdown("<h1 style='text-align: center; font-size:35px;'>Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.text("            by Syed Athaullah")

age = st.number_input("Age", 5, 99)
gender = st.number_input("Gender (0 = female, 1 = male)", 0, 1)
polyuria = st.number_input("Polyuria (0 = no, 1 = yes)", 0, 1)
polydipsia = st.number_input("Polydipsia (0 = no, 1 = yes)", 0, 1)
sudden_weight_loss = st.number_input("Sudden weight loss (0 = no, 1 = yes)", 0, 1)
weakness = st.number_input("Weakness (0 = no, 1 = yes)", 0, 1)
polyphagia = st.number_input("Polyphagia (0 = no, 1 = yes)", 0, 1)
genital_thrush = st.number_input("Genital thrush (0 = no, 1 = yes)", 0, 1)
visual_blurring = st.number_input("Visual blurring (0 = no, 1 = yes)", 0, 1)
itching = st.number_input("Itching (0 = no, 1 = yes)", 0, 1)
irritability = st.number_input("Irritability (0 = no, 1 = yes)", 0, 1)
delayed_healing = st.number_input("Delayed healing (0 = no, 1 = yes)", 0, 1)
partial_paresis = st.number_input("Partial paresis (0 = no, 1 = yes)", 0, 1)
muscle_stiffness = st.number_input("Muscle stiffness (0 = no, 1 = yes)", 0, 1)
alopecia = st.number_input("Alopecia (0 = no, 1 = yes)", 0, 1)
obesity = st.number_input("Obesity (0 = no, 1 = yes)", 0, 1)

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
