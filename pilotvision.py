import os
import streamlit as st
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Load data from local- v1
# @st.cache_data
# def load_data():
    # fixdataEcam = pd.read_csv("D:\\PilotVision_Proj\\fixdataEcam.csv")  # Update with your actual path
    # fixdataPfd = pd.read_csv("D:\\PilotVision_Proj\\fixdataPfd.csv")    #      ""
    # fixdataEfis = pd.read_csv("D:\\PilotVision_Proj\\fixdataEfis.csv")  #      ""
     #return fixdataEcam, fixdataPfd, fixdataEfis

# fixdataEcam, fixdataPfd, fixdataEfis = load_data()

# Load data from Azure Blob Storage - v2
# Set up connection to Azure Blob Storage
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set or is empty.")

# Initializing BlobServiceClient with connection string
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Function to load a CSV from Azure Blob Storage
def load_data_from_blob(blob_name):
    container_name = "pilotvisiondata" 
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    return pd.read_csv(StringIO(data.decode('utf-8')))  # Adjust encoding as needed

# Example usage in the Streamlit app
st.title("PilotVision Data Analysis")

# Load datasets from Azure Blob Storage
fixdataEcam = load_data_from_blob("fixdataEcam.csv")
fixdataPfd = load_data_from_blob("fixdataPfd.csv")
fixdataEfis = load_data_from_blob("fixdataEfis.csv")

# Data Availability Check
def verify_data_availability(data, name):
    if data.empty:
        st.warning(f"{name} data is not available or failed to load.")
    else:
        st.success(f"{name} data loaded successfully with {len(data)} records.")

verify_data_availability(fixdataEcam, "ECAM")
verify_data_availability(fixdataPfd, "PFD")
verify_data_availability(fixdataEfis, "EFIS")


# Data upload status check
def load_data_from_blob_safe(blob_name):
    try:
        return load_data_from_blob(blob_name)
    except Exception as e:
        st.error(f"Could not load {blob_name}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

fixdataEcam = load_data_from_blob_safe("fixdataEcam.csv")
fixdataPfd = load_data_from_blob_safe("fixdataPfd.csv")
fixdataEfis = load_data_from_blob_safe("fixdataEfis.csv")


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
