import pandas as pd

# Load the CSV files
fixdataEcam = pd.read_csv("D:\\PilotVision_Proj\\fixdataEcam.csv")
fixdataEfis = pd.read_csv("D:\\PilotVision_Proj\\fixdataEfis.csv")
fixdataPfd = pd.read_csv("D:\\PilotVision_Proj\\fixdataPfd.csv")

# Display column names and first few rows for each dataset
print("ECAM Data Columns:", fixdataEcam.columns)
print(fixdataEcam.head())
print("EFIS Data Columns:", fixdataEfis.columns)
print(fixdataEfis.head())
print("PFD Data Columns:", fixdataPfd.columns)
print(fixdataPfd.head())
