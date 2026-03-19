# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1) Load the weather dataset using pandas.
2) Preprocess the data by handling missing values and sorting by time.
3) Select features and create lag variables for temperature and PM2.5.
4) Train Random Forest models to predict temperature and PM2.5 and save the models.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by:Gowthaman Ak 
RegisterNumber:212225240043  
*/
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("Original rows:", len(df))

# Only drop if target missing
df = df.dropna(subset=['tem', 'pm2_5'])

# Fill feature columns instead of dropping
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['co2'] = df['co2'].fillna(df['co2'].mean())

# Sort by time
df = df.sort_values('time')

# Create lag features
df['Temp_Lag1'] = df['tem'].shift(1)
df['PM_Lag1'] = df['pm2_5'].shift(1)

# Only remove first row created by shift
df = df.iloc[1:]

print("Rows after preprocessing:", len(df))

# Features
X = df[['hum', 'pressure', 'wind_speed', 'co2',
        'Temp_Lag1', 'PM_Lag1']]

y_temp = df['tem']
y_pm = df['pm2_5']

print("Training samples:", len(X))

# Train models
model_temp = RandomForestRegressor(n_estimators=300, random_state=42)
model_pm = RandomForestRegressor(n_estimators=300, random_state=42)

model_temp.fit(X, y_temp)
model_pm.fit(X, y_pm)

# Save models
joblib.dump(model_temp, "temperature_model.pkl")
joblib.dump(model_pm, "pm25_model.pkl")

print("Models trained and saved successfully!")
```

## Output:
![WhatsApp Image 2026-03-19 at 11 24 59 AM](https://github.com/user-attachments/assets/02a7dd1a-f83c-499f-bac7-9c9314eb943b)
![WhatsApp Image 2026-03-19 at 11 24 59 AM](https://github.com/user-attachments/assets/7f724a40-5bf9-4989-bc7a-9fcb9a13fc1d)
![WhatsApp Image 2026-03-19 at 11 25 45 AM](https://github.com/user-attachments/assets/a96a7e19-5865-4ded-87f2-33d31d0a80f7)
![WhatsApp Image 2026-03-19 at 11 25 45 AM](https://github.com/user-attachments/assets/0eacfcd6-0f89-4457-8e38-7f9046f976a8)
![WhatsApp Image 2026-03-19 at 11 26 30 AM](https://github.com/user-attachments/assets/22b0104f-0b5a-47bd-8381-5754fce8b797)


## Result:
