from explore_ozone import data_subset
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def prepare_features(data):
    """Add cyclical month features to the dataset"""
    data['month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    return data

def train_multi_factor_model(data):
    """Train gradient boosting model and detect anomalies"""
    features = ['Pressure', 'Latitude', 'Longitude', 'month_sin', 'month_cos']
    X_multi = data[features]
    y_multi = data['Ozone_ppbv']

    print("New feature matrix shape:", X_multi.shape)
    print("Features:", X_multi.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
    multi_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    multi_model.fit(X_train, y_train)
    
    data['predicted_ozone_multi'] = multi_model.predict(X_multi)
    data['residual_multi'] = data['Ozone_ppbv'] - data['predicted_ozone_multi']
    
    # Detect anomalies using IQR method
    Q1_multi = data['residual_multi'].quantile(0.25)
    Q3_multi = data['residual_multi'].quantile(0.75)
    IQR_multi = Q3_multi - Q1_multi

    lower_bound_multi = Q1_multi - 1.5 * IQR_multi
    upper_bound_multi = Q3_multi + 1.5 * IQR_multi

    data['is_anomaly_multi'] = (data['residual_multi'] < lower_bound_multi) | (data['residual_multi'] > upper_bound_multi)
    
    print("Anomalies found by multi-factor model:", data['is_anomaly_multi'].sum())
    
    return data, multi_model

def plot_geographic_anomalies(data):
    """Plot geographic distribution of anomalies"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Longitude'], data['Latitude'], alpha=0.1, s=2, color='grey', label='Normal')
    plt.scatter(data[data['is_anomaly_multi']]['Longitude'],
                data[data['is_anomaly_multi']]['Latitude'],
                alpha=0.5, s=10, color='red', label='Anomaly (Multi-Factor)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of Anomalies (Multi-Factor Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/anomalies_map_multi_factor.png')
    plt.show()

def plot_anomaly_rate_heatmap(data):
    """Create heatmap of anomaly rate by geographic location"""
    lat_bins = np.linspace(data['Latitude'].min(), data['Latitude'].max(), 20)
    lon_bins = np.linspace(data['Longitude'].min(), data['Longitude'].max(), 20)

    grouped = data.groupby([pd.cut(data['Latitude'], lat_bins), 
                           pd.cut(data['Longitude'], lon_bins)])
    anomaly_rate = grouped['is_anomaly_multi'].mean() * 100

    plt.figure(figsize=(12, 8))
    plt.imshow(anomaly_rate.unstack(), aspect='auto', cmap='Reds', 
               extent=[lon_bins.min(), lon_bins.max(), lat_bins.min(), lat_bins.max()])
    plt.colorbar(label='Anomaly Rate (%)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Heatmap of Anomaly Rate')
    plt.tight_layout()
    plt.savefig('figures/anomaly_rate_heatmap.png')
    plt.show()

def plot_monthly_anomaly_rate(data):
    """Plot anomaly rate by month"""
    monthly_anomaly_rate = data.groupby('Month')['is_anomaly_multi'].mean() * 100

    plt.figure(figsize=(10, 5))
    monthly_anomaly_rate.plot(kind='bar')
    plt.xlabel('Month')
    plt.ylabel('Anomaly Rate (%)')
    plt.title('Anomaly Rate by Month')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('figures/anomaly_rate_by_month.png')
    plt.show()

def show_top_anomalies(data, n=10):
    """Display top anomalies"""
    top_anomalies = data.nlargest(n, 'residual_multi')
    
    print(f"\n[TOP {n} HIGH-OZONE ANOMALIES]")
    print(top_anomalies[['Year', 'Month', 'Pressure', 'Latitude', 'Longitude', 
                         'Ozone_ppbv', 'predicted_ozone_multi', 'residual_multi']].to_string())

def plot_latitude_month_heatmap(data):
    """Create heatmap of anomaly rate by latitude and month"""
    pivot_table = data.groupby(['Latitude', 'Month'])['is_anomaly_multi'].mean().unstack()

    plt.figure(figsize=(14, 8))
    im = plt.imshow(pivot_table.values, 
                    cmap='RdBu_r',
                    aspect='auto', 
                    extent=[1, 12, data['Latitude'].max(), data['Latitude'].min()])

    cbar = plt.colorbar(im, label='Anomaly Rate')
    cbar.set_label('Anomaly Rate (Fraction of Measurements)', rotation=270, labelpad=20)

    plt.xlabel('Month')
    plt.ylabel('Latitude')
    plt.title('Anomaly Rate by Latitude and Month', fontsize=14)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.yticks(np.arange(32, 54, 2))

    X, Y = np.meshgrid(range(1, 13), pivot_table.index)
    plt.contour(X, Y, pivot_table.values, levels=5, colors='black', linewidths=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/anomaly_heatmap_latitude_month.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_monthly_facet_grid(data):
    """Create facet grid of monthly anomaly maps"""
    nmonths = 12
    ncols = 4
    nrows = math.ceil(nmonths / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))
    axes = axes.flatten()

    for i, month in enumerate(range(1, 13)):
        month_data = data[data['Month'] == month]
        
        hexbin = axes[i].hexbin(month_data['Longitude'], month_data['Latitude'], 
                                C=month_data['is_anomaly_multi'], 
                                reduce_C_function=np.mean,
                                gridsize=30, 
                                cmap='RdBu_r')
        
        axes[i].set_title(f'Month: {month}')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        
        plt.colorbar(hexbin, ax=axes[i], label='Anomaly Rate')

    for j in range(nmonths, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/anomaly_facet_grid_monthly_maps.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_3d_anomaly_distribution(data):
    """Create 3D plot of anomaly distribution"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    plot_sample = data.sample(n=10000, random_state=42)

    scatter = ax.scatter(plot_sample['Longitude'], 
                         plot_sample['Latitude'], 
                         plot_sample['Month'],
                         c=plot_sample['is_anomaly_multi'], 
                         cmap='coolwarm',
                         alpha=0.6,
                         s=10)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Month')
    ax.set_title('3D Anomaly Distribution: Latitude, Longitude, Month')
    plt.colorbar(scatter, label='Is Anomaly (0=Normal, 1=Anomaly)')

    ax.view_init(elev=20, azim=45)

    plt.savefig('figures/3d_anomaly_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()