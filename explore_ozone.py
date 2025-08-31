import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def load_and_preprocess_data(filepath):
    """Load data and filter by pressure range"""
    data = pd.read_csv(filepath)
    data_subset = data[(data['Pressure'] >= 300) & (data['Pressure'] <= 900)].copy()
    print(f"Loaded data with {len(data_subset)} rows after pressure filtering")
    return data_subset

def train_linear_model(X, y):
    """Train a linear regression model and return the model object"""
    model = LinearRegression()
    model.fit(X, y)
    return model

def analyze_residuals(data, model, feature_col='Pressure', target_col='Ozone_ppbv'):
    """Calculate predictions and residuals, and perform residual analysis"""
    # Make predictions
    X = data[[feature_col]]
    data['predicted_ozone'] = model.predict(X)
    
    # Calculate residuals
    data['residual'] = data[target_col] - data['predicted_ozone']
    
    # Print model coefficients and residual statistics
    print(f"\n[MODEL COEFFICIENTS]")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    print(f"\n[RESIDUAL ANALYSIS]")
    print(data['residual'].describe())
    
    return data

def plot_residual_histogram(data, save_path='figures/residuals_histogram.png'):
    """Plot histogram of residuals"""
    plt.figure(figsize=(10, 6))
    plt.hist(data['residual'], bins=100)
    plt.xlabel('Residual (Actual Ozone - Predicted Ozone)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_model_fit(data, model, feature_col='Pressure', target_col='Ozone_ppbv', 
                  save_path='figures/linear_model_fit.png'):
    """Plot actual data vs model prediction with smooth prediction line"""
    # Create pressure range for smooth prediction line
    pressure_min = data[feature_col].min()
    pressure_max = data[feature_col].max()
    pressure_range = np.linspace(pressure_min, pressure_max, 100).reshape(-1, 1)
    predicted_ozone_range = model.predict(pressure_range)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data[target_col], data[feature_col], alpha=0.01, s=5, 
                label='Actual Data', color='red')
    plt.plot(predicted_ozone_range, pressure_range, linewidth=3, 
             color='blue', label='Linear Model Prediction')
    
    plt.gca().invert_yaxis()  # Puts high altitude (low pressure) at the top
    plt.xlabel('Ozone (ppbv)')
    plt.ylabel('Pressure (hPa)')
    plt.title('Linear Model of Tropospheric Ozone vs. Pressure')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()