import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Load data
@st.cache_data
def load_data():
    fixdataEcam = pd.read_csv("D:\\PilotVision_Proj\\fixdataEcam.csv")  # Update with your actual path
    fixdataPfd = pd.read_csv("D:\\PilotVision_Proj\\fixdataPfd.csv")    #      ""
    fixdataEfis = pd.read_csv("D:\\PilotVision_Proj\\fixdataEfis.csv")  #      ""
    return fixdataEcam, fixdataPfd, fixdataEfis

fixdataEcam, fixdataPfd, fixdataEfis = load_data()

# Combine data for different surfaces
ecam_pipe = fixdataEcam[["duration [ms]"]].rename(columns={"duration [ms]": "ECAM"})
pfd_pipe = fixdataPfd[["duration [ms]"]].rename(columns={"duration [ms]": "PFD"})
efis_pipe = fixdataEfis[["duration [ms]"]].rename(columns={"duration [ms]": "EFIS"})
all_surfaces = pd.concat([ecam_pipe, pfd_pipe, efis_pipe], axis=1)

st.write("### Combined Data for Different Surfaces")
st.write(all_surfaces.head())

# Model Training and Evaluation
def train_and_evaluate(surface, model_type):
    data = {"ECAM": fixdataEcam, "PFD": fixdataPfd, "EFIS": fixdataEfis}[surface]
    X = data[["duration [ms]"]]
    y = data["fixation x [normalized]"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = Normalizer()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = LinearRegression() if model_type == 'Linear Regression' else MLPRegressor(activation='logistic', solver='lbfgs', max_iter=500000)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = model.score(X, y)
    return rmse_train, rmse_test, r2

st.sidebar.title("Model Training and Evaluation")
surface = st.sidebar.selectbox('Select Surface', ['ECAM', 'PFD', 'EFIS'])
model_type = st.sidebar.selectbox('Select Model', ['Linear Regression', 'ANN'])
if st.sidebar.button('Run Model'):
    rmse_train, rmse_test, r2 = train_and_evaluate(surface, model_type)
    st.write(f"### Results for {model_type} on {surface}")
    st.write(f'Training RMSE: {rmse_train}')
    st.write(f'Testing RMSE: {rmse_test}')
    st.write(f'R²: {r2}')

def display_dashboard():
    st.write("### Dashboard")
    st.write('Dashboard Content Here')
    # Additional dashboard logic goes here

if st.sidebar.button('Show Dashboard'):
    display_dashboard()
