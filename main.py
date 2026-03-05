import pandas as pd
import gdown
import os

from src.data_cleaning import run_cleaning
from src.feature_engineering import run_features
from src.model_training import run_modeling

file_id = '1C0dxQBvT92qE1dVLtnhtbzvM4JHu3sF6'
output = 'data/ML_Project_Raw_Dataset.csv'

if not os.path.exists('data'): os.makedirs('data')
if not os.path.isfile(output):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

df_raw = pd.read_csv(output, encoding='latin1', low_memory=False)

print("--- Starting ML Pipeline ---")

df_cleaned = run_cleaning(df_raw)
df_engineered = run_features(df_cleaned)
final_results = run_modeling(df_engineered)

print("\n--- Final Model Comparison ---")
print(final_results)

if not os.path.exists('results'): os.makedirs('results')
final_results.to_csv('results/model_metrics.csv', index=False)