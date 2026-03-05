import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import gdown

def run_cleaning(df):
    # Initial Setup
    warnings.filterwarnings("ignore")

    file_id = '1C0dxQBvT92qE1dVLtnhtbzvM4JHu3sF6' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output_folder = 'data'
    output_file = os.path.join(output_folder, 'ML_Project_Raw_Dataset.csv')

    # Create data directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download the file if it's not already there
    if not os.path.isfile(output_file):
        print("Fetching dataset from Google Drive...")
        gdown.download(url, output_file, quiet=False)
    else:
        print("Dataset already present in /data folder.")

    # Load Data
    df = pd.read_csv(output_file, encoding='latin1', low_memory=False)

    # Column Selection and Initial Drop
    useful_columns = [
        'price', 'living_space', 'bedroom_number', 'bathroom_number', 
        'room_number', 'parking_number', 'property_type', 'city', 
        'state', 'built_year', 'furnished', 'new_home'
    ]
    df = df[useful_columns].copy()
    df = df.drop_duplicates()
    df = df.drop(columns=['parking_number', 'new_home', 'room_number'])

    # Preliminary Filtering
    df = df[df['property_type'] != 'Land']
    df = df.dropna(subset=['price'])

    # Type Conversion and Imputation
    cols_to_numeric = ['bathroom_number', 'bedroom_number', 'living_space', 'built_year']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=['city', 'state', 'property_type'])
    df['furnished'] = df['furnished'].fillna('Unknown')
    df = df.drop_duplicates()

    # Price Outlier Removal (Manual & IQR)
    df = df[df['price'] > 500_000]
    df = df[df['price'] < 100_000_000]

    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

    # Physical Outlier Removal
    df = df[df['living_space'].between(10, 10000)]
    df = df[df['bedroom_number'].between(1, 15)]
    df = df[df['bathroom_number'].between(1, 15)]

    print("Cleaning complete.")
    return df
