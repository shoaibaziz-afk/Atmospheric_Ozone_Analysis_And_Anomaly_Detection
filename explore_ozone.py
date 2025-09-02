import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from data_loading import df

X = df['Pressure']
Y = df['Ozone_ppbv']

def train_linear_model(X, Y):
    """Train a linear regression model and return the model object"""
    model = LinearRegression()
    model.fit(X, Y)
    print(model)
    return model

def analyze_residuals(df, model, feature_col='Pressure', target_col='Ozone_ppbv'):
    """Calculate predictions and residuals, and perform residual analysis"""
    X = df[[feature_col]]
    df['predicted_ozone'] = model.predict(X)
    
    df['residual'] = df[target_col] - df['predicted_ozone']
    
    print(f"\n[MODEL COEFFICIENTS]")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    print(f"\n[RESIDUAL ANALYSIS]")
    print(df['residual'].describe())
    
    return df

def plot_residual_histogram(df, save_path='figures/residuals_histogram.png'):
    """Plot histogram of residuals"""
    plt.figure(figsize=(10, 6))
    plt.hist(df['residual'], bins=100)
    plt.xlabel('Residual (Actual Ozone - Predicted Ozone)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_model_fit(df, model, feature_col='Pressure', target_col='Ozone_ppbv', 
                  save_path='figures/linear_model_fit.png'):
    """Plot actual df vs model prediction with smooth prediction line"""
    # Create pressure range for smooth prediction line
    pressure_min = df[feature_col].min()
    pressure_max = df[feature_col].max()
    pressure_range = np.linspace(pressure_min, pressure_max, 100).reshape(-1, 1)
    predicted_ozone_range = model.predict(pressure_range)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[target_col], df[feature_col], alpha=0.01, s=5, 
                label='Actual df', color='red')
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