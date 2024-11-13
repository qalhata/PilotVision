# PilotVision
A view into pilot performance using eye-tracking data and the broader focus on flight data insights

# This is a Streamlit Data Visualization App

This is an open-source Streamlit app for visualizing Pilot in-flight eye tracking data data.
This is a deployment built to work with a Azure blob storage container

## Prerequisites
- Azure account with storage and web app permissions
- Python 3.8+
- [Streamlit](https://streamlit.io/)

## Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/qalhata/PilotVision
   cd repo-name


## Don't forget to do the following once cloned
pip install -r requirements.txt

You can also download the files to you azure blob storage and then

## Set up your env variables for Axure
export AZURE_STORAGE_CONNECTION_STRING="YourAzureConnectionString"

## Run the App with: 

streamlit run pilotvision.py

