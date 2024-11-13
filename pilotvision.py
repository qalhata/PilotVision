import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Set Streamlit configurations
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Load environment variables from .env file
load_dotenv()

# Set up connection to Azure Blob Storage
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set or is empty.")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Function to load a CSV from Azure Blob Storage
def load_data_from_blob(blob_name):
    container_name = "pilotvisiondata" 
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    return pd.read_csv(StringIO(data.decode('utf-8')))  # Adjust encoding as needed

# Load datasets from Azure Blob Storage
fixdataEcam = load_data_from_blob("fixdataEcam.csv")
fixdataPfd = load_data_from_blob("fixdataPfd.csv")
fixdataEfis = load_data_from_blob("fixdataEfis.csv")

# Verify data availability
def verify_data_availability(data, name):
    if data.empty:
        st.warning(f"{name} data is not available or failed to load.")
    else:
        st.success(f"{name} data loaded successfully with {len(data)} records.")

verify_data_availability(fixdataEcam, "ECAM")
verify_data_availability(fixdataPfd, "PFD")
verify_data_availability(fixdataEfis, "EFIS")

# Plotting Functions
def plot_fixation_distribution(data, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['duration [ms]'], kde=True, color="skyblue")
    plt.title(title)
    plt.xlabel("Fixation Duration (ms)")
    plt.ylabel("Frequency")
    st.pyplot()

def plot_fixation_time_series(data, title):
    data['start_time'] = pd.to_datetime(data['start timestamp [ns]'], unit='ns')
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=data['start_time'], y=data['duration [ms]'], color="blue")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Fixation Duration (ms)")
    st.pyplot()

def plot_fixation_heatmap(data, title):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=data['fixation x [normalized]'], y=data['fixation y [normalized]'], cmap="Reds", shade=True, bw_adjust=0.5)
    plt.title(title)
    plt.xlabel("X Coordinate [Normalized]")
    plt.ylabel("Y Coordinate [Normalized]")
    st.pyplot()

# Dashboard Function for Dataset Analysis
def display_dashboard(dataset_name):
    if dataset_name == "ECAM":
        st.write("### ECAM Dataset Analysis")
        st.write("This analysis helps reveal pilot attention distribution on the Engine and Centralized Aircraft Monitoring (ECAM) display.")
        plot_fixation_distribution(fixdataEcam, "ECAM Fixation Duration Distribution")
        plot_fixation_time_series(fixdataEcam, "ECAM Fixation Duration Over Time")
        plot_fixation_heatmap(fixdataEcam, "ECAM Fixation Focus Heatmap")
        
    elif dataset_name == "EFIS":
        st.write("### EFIS Dataset Analysis")
        st.write("This analysis reveals how pilots allocate attention to key flight parameters on the Electronic Flight Instrument System (EFIS).")
        plot_fixation_distribution(fixdataEfis, "EFIS Fixation Duration Distribution")
        plot_fixation_time_series(fixdataEfis, "EFIS Fixation Duration Over Time")
        plot_fixation_heatmap(fixdataEfis, "EFIS Fixation Focus Heatmap")
        
    elif dataset_name == "PFD":
        st.write("### PFD Dataset Analysis")
        st.write("This analysis helps understand how pilots monitor essential flight indicators on the Primary Flight Display (PFD).")
        plot_fixation_distribution(fixdataPfd, "PFD Fixation Duration Distribution")
        plot_fixation_time_series(fixdataPfd, "PFD Fixation Duration Over Time")
        plot_fixation_heatmap(fixdataPfd, "PFD Fixation Focus Heatmap")

# Sidebar for dataset selection
st.sidebar.title("Data Selection and Model Training")
dataset_name = st.sidebar.selectbox("Choose a dataset to analyze", ["ECAM", "EFIS", "PFD"])

if st.sidebar.button('Show Dashboard'):
    display_dashboard(dataset_name)

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

surface = st.sidebar.selectbox('Select Surface for Model', ['ECAM', 'PFD', 'EFIS'])
model_type = st.sidebar.selectbox('Select Model Type', ['Linear Regression', 'ANN'])
if st.sidebar.button('Run Model'):
    rmse_train, rmse_test, r2 = train_and_evaluate(surface, model_type)
    st.write(f"### Results for {model_type} on {surface}")
    st.write(f'Training RMSE: {rmse_train}')
    st.write(f'Testing RMSE: {rmse_test}')
    st.write(f'R²: {r2}')