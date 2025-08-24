import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv('Receptor_western_NAmerica_ozone_obs_1994_2021_from900to300.csv')

print("\n[INFO] Generating time series plot...")
data_subset = data[(data['Pressure'] >= 300) & (data['Pressure'] <= 900)].copy()

# 1. DEFINE FEATURE (X) AND TARGET (Y)
X = data_subset[['Pressure']]  # Feature DataFrame
y = data_subset['Ozone_ppbv']   # Target Series

# 2. TRAIN THE LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X, y) # Train the model using our data (X, y)

print(f"Slope: {model.coef_[0]}")
print(f"Model intercept: {model.intercept_}")

# 3. MAKE PREDICTIONS ON THE SAME DATA
data_subset['predicted_ozone'] = model.predict(X) # Create predictions for all points

# 4. CALCULATE RESIDUALS (Actual Value - Predicted Value)
data_subset['residual'] = data_subset['Ozone_ppbv'] - data_subset['predicted_ozone']

# 5. ANALYZE THE RESIDUALS
print("\n[RESIDUAL ANALYSIS]")
print(data_subset['residual'].describe())

# 6. PLOT HISTOGRAM OF RESIDUALS
plt.figure(figsize=(10, 6))
plt.hist(data_subset['residual'], bins=100)
plt.xlabel('Residual (Actual Ozone - Predicted Ozone)')
plt.ylabel('Frequency')
plt.title('Distribution of Model Residuals')
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/residuals_histogram.png')
plt.show()

# 7. CREATE PRESSURE RANGE FOR SMOOTH PREDICTION LINE
pressure_range = np.linspace(data_subset['Pressure'].min(), data_subset['Pressure'].max(), 100).reshape(-1, 1)
predicted_ozone_range = model.predict(pressure_range)

# 8. PLOT ACTUAL DATA VS MODEL PREDICTION
plt.figure(figsize=(10, 6))
plt.scatter(data_subset['Ozone_ppbv'], data_subset['Pressure'], alpha=0.01, s=5, label='Actual Data', color='red')
plt.plot(predicted_ozone_range, pressure_range, linewidth=3, color='blue', label='Linear Model Prediction')

plt.gca().invert_yaxis()  # Puts high altitude (low pressure) at the top
plt.xlabel('Ozone (ppbv)')
plt.ylabel('Pressure (hPa)')
plt.title('Linear Model of Tropospheric Ozone vs. Pressure')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/linear_model_fit.png')
plt.show()