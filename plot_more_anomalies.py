from explore_ozone import data_subset
import matplotlib.pyplot as plt
import pandas as pd

def get_top_anomalies(data, n=200, residual_col='residual'):
    """Get the top N most extreme anomalies by absolute residual value"""
    top_anomalies = data.sort_values(by=residual_col, key=abs, ascending=False).head(n)
    
    print(f"\n[TOP {n} MOST EXTREME ANOMALIES]")
    print(top_anomalies[['Pressure', 'Ozone_ppbv', 'predicted_ozone', 'residual']].to_string())
    
    return top_anomalies

def plot_top_anomalies_context(data, top_anomalies, n=200, 
                              ozone_col='Ozone_ppbv', pressure_col='Pressure',
                              save_path='figures/top_anomalies_context.png'):
    """Plot top anomalies in context with all data points"""
    plt.figure(figsize=(10, 6))
    
    # Plot all points
    plt.scatter(data[ozone_col], data[pressure_col], 
                alpha=0.1, s=2, color='grey', label='Normal')
    
    # Highlight the top anomalies
    plt.scatter(top_anomalies[ozone_col], top_anomalies[pressure_col], 
                alpha=1, s=50, color='red', label=f"Top {n} anomalies")
    
    plt.gca().invert_yaxis()
    plt.xlabel('Ozone (ppbv)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f"Top {n} Anomalies in Context")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    print(f"[PLOT SAVED] Top anomalies context plot saved to: {save_path}")

def calculate_annual_anomaly_rate(data, anomaly_col='is_anomaly', year_col='Year'):
    """Calculate annual anomaly rates"""
    annual_anomalies = data[data[anomaly_col]].groupby(year_col).size()
    annual_total = data.groupby(year_col).size()
    
    # Calculate anomaly rate per year
    annual_rate = (annual_anomalies / annual_total) * 100  # Percentage
    
    return annual_rate, annual_anomalies, annual_total

def plot_annual_anomaly_rate(annual_rate, save_path='figures/anomaly_rate_by_year.png'):
    """Plot annual anomaly rate as bar chart"""
    plt.figure(figsize=(12, 5))
    annual_rate.plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Anomaly Rate (%)')
    plt.title('Percentage of Anomalous Measurements per Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    print(f"[PLOT SAVED] Annual anomaly rate plot saved to: {save_path}")
    
    # Print summary statistics
    print(f"\n[ANNUAL ANOMALY RATE SUMMARY]")
    print(f"Mean anomaly rate: {annual_rate.mean():.2f}%")
    print(f"Median anomaly rate: {annual_rate.median():.2f}%")
    print(f"Minimum anomaly rate: {annual_rate.min():.2f}% (Year {annual_rate.idxmin()})")
    print(f"Maximum anomaly rate: {annual_rate.max():.2f}% (Year {annual_rate.idxmax()})")