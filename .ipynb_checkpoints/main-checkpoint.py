import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
import joblib

html_temp = '''
    <div style="background-color: rgba(25,25,112,0.0); padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Weather Predictor</h1></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

# Load the exported models
adaboost_model = joblib.load('AdaBoostClassifier_model.joblib')
rf_model = joblib.load('RandomForestClassifier_model.joblib')
nb_model = joblib.load('GaussianNB_model.joblib')
knn_model = joblib.load('KNeighborsClassifier_model.joblib')
mlp_model = joblib.load('MLPClassifier_model.joblib')
mnb_model = joblib.load('MultinomialNB_model.joblib')

models = {
    "AdaBoost Classifier": adaboost_model,
    "Random Forest Classifier": rf_model,
    "Gaussian Naive Bayes": nb_model,
    "K Nearest Neighbors Classifier": knn_model,
    "MLP Classifier": mlp_model,
    "Multinomial Naive Bayes": mnb_model,
}

model = st.selectbox("Please select your model from the following options", list(models.keys()))

if model == "Please Select":
    st.error("Please select a valid model")
    st.stop()

data = []
data.append(st.sidebar.date_input('Enter Date'))
data.append(st.sidebar.slider('Max Temperatuer',15,40,20))
data.append(st.sidebar.slider('Min Temperatuer',15,40,20))
data.append(st.sidebar.slider('Pressure', 700, 800, 760))
data.append(st.sidebar.number_input('Humidity'))
data.append(st.sidebar.number_input('Wind Speed'))

origin = datetime.datetime(2017, 4, 1)
data[0] = pd.to_datetime(data[0]).strftime('%Y-%m-%d')
data = pd.DataFrame(data).transpose()
data.columns = ['Date','maxtemp','mintemp','Pressure', 'Humidity', 'Wind Speed']
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

del data['Date']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

if st.button('Predict'):
    ans_model = models[model]
    ans = ans_model.predict(data_scaled)
    st.info('Answer: {}'.format(ans[0]))
