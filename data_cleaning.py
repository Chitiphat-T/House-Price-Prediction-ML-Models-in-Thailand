import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

# Initial Setup
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'ML_Project_Raw_Dataset.csv')

# Load Data
df = pd.read_csv(data_path, encoding='latin1', low_memory=False)

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

print(f"Max price after cleaning: {df['price'].max():,.2f}")
print(f"Rows remaining after cleanup: {len(df)}")
