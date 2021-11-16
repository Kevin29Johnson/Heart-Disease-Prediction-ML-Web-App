import pandas as pd
import streamlit as st
#import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
st.write(
    """** Heart Disease Prediction App**"""
)
st.sidebar.header('User Input Features')


def user_input_features():
    age = st.sidebar.slider("age", 21, 81, 29)
    sex = st.sidebar.selectbox("(1=Male, 0=Female)", ["1", "0"])
    cp = st.sidebar.selectbox(
        "Chest pain(1 = typical angina ,2 = atypical angina,3 = non — anginal pain,4 = asymptotic)", ["1", "2", "3", "4"])
    trestbps = st.sidebar.slider("resting blood pressure", 100, 400, 110)
    chol = st.sidebar.slider("chol", 100, 400, 110)
    fbs = st.sidebar.selectbox(
        "Fasting blood sugar (1=fbs>120mg/dl, 0=fbs<120 mg/dl)", ["1", "0"])
    restecg = st.sidebar.selectbox(
        "Resting ECG (0 = normal ,1 = having ST-T wave abnormality,2 = left ventricular hyperthrophy)", ["0", "1", "2"])
    thalach = st.sidebar.slider("maximum heart rate achieved", 100, 200, 150)
    exang = st.sidebar.selectbox(
        "exercise induced angina(1=yes , 0=no)", ["1", "0"])
    oldpeak = st.sidebar.slider(
        "ST depression induced by exercise relative to rest", 0.0, 5.0, 3.4)
    slope = st.sidebar.slider(
        "the slope of the peak exercise ST segment", 0, 2, 1)
    ca = st.sidebar.slider(
        "number of major vessels (0-3) colored by flourosopy", 0, 3, 1)
    thal = st.sidebar.slider(
        "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect", 0.0, 10.0, 3.0)
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
st.write(df)
heart_raw = pd.read_csv("heart.csv")
X = heart_raw.drop(['target'], axis=1)
Y = heart_raw.target
model = RandomForestClassifier()
model.fit(X, Y)

prediction = model.predict(df)
st.subheader('Prediction :')
df1 = pd.DataFrame(prediction, columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(df)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)
