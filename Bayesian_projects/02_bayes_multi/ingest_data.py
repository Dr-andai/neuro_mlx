import pandas as pd
import numpy as np
import mlx.core as mx

# data from ML repo
data = pd.read_csv('beijing_data.csv').dropna()

# print(data['pm2.5'].describe())
""""
count    41757.000000
mean        98.613215
std         92.050387
min          0.000000
25%         29.000000
50%         72.000000
75%        137.000000
max        994.000000
"""

# get PM values
pm_values = data['pm2.5'].values

data['pm2.5'].describe()

# Map pm2.5 values to health risk categories
"""
class 0: baseline
class 1: mild hazard
class 2: elevated hazard
class 3: severe stress
"""
conditions = [
    (pm_values <= 12.0),
    (pm_values > 12.0) & (pm_values <= 35.4),
    (pm_values > 12.0) & (pm_values <= 55.4),
    (pm_values > 55.4),
]

choices = [0, 1, 2, 3]
y_raw = np.select(conditions, choices, default=3).astype(np.int32)

# Process structural inputs
# Need to drop categorical targets
X_df = data.drop(columns=['No','year','month','day','hour','pm2.5'])
X_df = pd.get_dummies(X_df, columns=['cbwd'], drop_first=True)
X_raw = X_df.values.astype(np.float32)

# Normalise and scale features
X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
X_scaled = (X_raw - X_mean) / X_std
"""
Subtracting the mean and dividing by the standard deviation forces every 
feature to sit on a unified scale centered around 0, with a variance of 1
"""

# Array partitioning
split = int(len(X_scaled) * 0.8)
X_train, X_test = mx.array(X_scaled[:split]), mx.array(X_scaled[split:])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")