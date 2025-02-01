import pandas as pd
import numpy as np
import pickle
import streamlit as st 
import tensorflow


st.title("Salary Estimator")

credit = st.number_input('Credit Score')
age = st.slider('Age', 10, 98, 20)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
products = st.slider('No. of Products', 0, 10)
card = st.selectbox('HasCrCard', [0, 1])
member = st.selectbox('IsActiveMember', [0, 1])
exited = st.selectbox('Exited', [0, 1])
geography = st.selectbox('Geography', ['Germany', 'France', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])

if geography == 'Germany':
    Geography_Germany = 1
    Geography_Spain = 0
elif geography == 'Spain':
    Geography_Germany = 0
    Geography_Spain = 1
else:
    Geography_Germany = 0
    Geography_Spain = 0
    
if gender == 'Male':
    Gender_Male = 1
else:
    Gender_Male = 0
    

input_data = pd.DataFrame({
    'CreditScore' : [credit], 
    'Age' : [age] , 
    'Tenure' : [tenure], 
    'Balance' : [balance], 
    'NumOfProducts' : [products], 
    'HasCrCard' : [card],
    'IsActiveMember' : [member], 
    'Exited' : [exited], 
    'Geography_Germany' : [Geography_Germany],
    'Geography_Spain' : [Geography_Spain], 
    'Gender_Male': [Gender_Male]
})

with open('scaler.pkl', 'br') as file:
    scaler = pickle.load(file)
    
scaled_data = scaler.transform(input_data)

model = tensorflow.keras.models.load_model('salary.keras')
predictions = model.predict(scaled_data)

st.write(f"The estimated salary is {round(predictions[0][0], 2)}")