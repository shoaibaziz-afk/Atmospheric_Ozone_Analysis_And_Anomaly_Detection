from data_loading import df
import matplotlib.pyplot as plt
import pandas as pd

def calculate_anomaly_thresholds(df, residual_col='residual'):
    """Calculate IQR-based anomaly thresholds"""
    Q1 = df[residual_col].quantile(0.25)
    Q3 = df[residual_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\n[ANOMALY THRESHOLD CALCULATION]")
    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound (anomaly if residual < {lower_bound:.2f})")
    print(f"Upper bound (anomaly if residual > {upper_bound:.2f})")
    
    return lower_bound, upper_bound, Q1, Q3, IQR

def detect_anomalies(df, lower_bound, upper_bound, residual_col='residual'):
    """Detect anomalies based on thresholds and add to dfFrame"""
    df['is_anomaly'] = (df[residual_col] < lower_bound) | (df[residual_col] > upper_bound)
    
    anomaly_count = df['is_anomaly'].sum()
    total_count = len(df)
    
    print(f"\n[ANOMALY COUNT] {anomaly_count} anomalies found out of {total_count} total points ({anomaly_count/total_count:.2%})")
    
    return df, anomaly_count

def save_anomalies_to_csv(df, filepath='ozone_df_with_anomalies.csv'):
    """Save dfFrame with anomaly flags to CSV"""
    df.to_csv(filepath, index=False)
    print(f"\n[SAVED] Anomaly df saved to: {filepath}")
    return filepath

def plot_anomalies(df, ozone_col='Ozone_ppbv', pressure_col='Pressure', 
                  anomaly_col='is_anomaly', save_path='figures/anomalies_detected.png'):
    """Plot anomalies vs normal points"""
    plt.figure(figsize=(10, 6))

    # Plot all normal points in grey
    plt.scatter(df[~df[anomaly_col]][ozone_col],
                df[~df[anomaly_col]][pressure_col],
                alpha=0.1, s=5, color='grey', label='Normal')

    # Plot all anomaly points in red
    plt.scatter(df[df[anomaly_col]][ozone_col],
                df[df[anomaly_col]][pressure_col],
                alpha=0.7, s=20, color='red', label='Anomaly')

    plt.gca().invert_yaxis()
    plt.xlabel('Ozone (ppbv)')
    plt.ylabel('Pressure (hPa)')
    plt.title('Anomalies in Tropospheric Ozone df (IQR Method)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    print(f"[PLOT SAVED] Anomaly plot saved to: {save_path}")