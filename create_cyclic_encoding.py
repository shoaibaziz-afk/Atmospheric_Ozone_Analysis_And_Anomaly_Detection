from explore_ozone import data_subset
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data_subset['month_sin'] = np.sin(2 * np.pi * data_subset['Month'] / 12)
data_subset['month_cos'] = np.cos(2 * np.pi * data_subset['Month'] / 12)

features = ['Pressure', 'Latitude', 'Longitude', 'month_sin', 'month_cos']
X_multi = data_subset[features]
y_multi = data_subset['Ozone_ppbv']

print("New feature matrix shape:", X_multi.shape)
print("Features:", X_multi.columns.tolist())


X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size = 0.2, random_state = 42)
multi_model = GradientBoostingRegressor(n_estimators = 100, random_state = 42)
multi_model.fit(X_train, y_train)
data_subset['predicted_ozone_multi'] = multi_model.predict(X_multi)
data_subset['residual_multi'] = data_subset['Ozone_ppbv'] - data_subset['predicted_ozone_multi']

Q1_multi = data_subset['residual_multi'].quantile(0.25)
Q3_multi = data_subset['residual_multi'].quantile(0.75)
IQR_multi = Q3_multi - Q1_multi

lower_bound_multi = Q1_multi - 1.5 * IQR_multi
upper_bound_multi = Q3_multi + 1.5 * IQR_multi

data_subset['is_anomaly_multi'] = (data_subset['residual_multi'] < lower_bound_multi) | (data_subset['residual_multi'] > upper_bound_multi)

# Compare the results
#print("Anomalies found by simple model:", data_subset['is_anomaly'].sum())
print("Anomalies found by multi-factor model:", data_subset['is_anomaly_multi'].sum())


plt.figure(figsize=(10, 6))
plt.scatter(data_subset['Longitude'], data_subset['Latitude'], alpha=0.1, s=2, color='grey', label='Normal')
plt.scatter(data_subset[data_subset['is_anomaly_multi']]['Longitude'],
            data_subset[data_subset['is_anomaly_multi']]['Latitude'],
            alpha=0.5, s=10, color='red', label='Anomaly (Multi-Factor)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of Anomalies (Multi-Factor Model)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/anomalies_map_multi_factor.png')
plt.show()


# Anomalies clustered in specific geographic regions?"

# 1. Bin the data by Latitude and Longitude and calculate the anomaly RATE
# This shows us if certain areas are more prone to anomalies

# Create a 2D grid (histogram) of anomaly rate
lat_bins = np.linspace(data_subset['Latitude'].min(), data_subset['Latitude'].max(), 20)
lon_bins = np.linspace(data_subset['Longitude'].min(), data_subset['Longitude'].max(), 20)

# Group data into the bins and calculate the percentage of points that are anomalous in each bin
grouped = data_subset.groupby([pd.cut(data_subset['Latitude'], lat_bins), 
                               pd.cut(data_subset['Longitude'], lon_bins)])
anomaly_rate = grouped['is_anomaly_multi'].mean() * 100  # Percentage

# Plot a heatmap of the anomaly rate
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

# Seasonal anomalies

# 2. Plot anomaly rate by month to see if there's a seasonal pattern
# Use the original 'Month' column for the x-axis

# Calculate the anomaly rate for each month
monthly_anomaly_rate = data_subset.groupby('Month')['is_anomaly_multi'].mean() * 100

plt.figure(figsize=(10, 5))
monthly_anomaly_rate.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Anomaly Rate (%)')
plt.title('Anomaly Rate by Month')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/anomaly_rate_by_month.png')
plt.show()

# Worst Anomalies

# 3. Examine the top 10 most extreme anomalies in detail
top_anomalies = data_subset.nlargest(10, 'residual_multi')  # For high ozone anomalies
# For low anomalies: .nsmallest(10, 'residual_multi')

print("\n[TOP 10 HIGH-OZONE ANOMALIES]")
print(top_anomalies[['Year', 'Month', 'Pressure', 'Latitude', 'Longitude', 
                     'Ozone_ppbv', 'predicted_ozone_multi', 'residual_multi']].to_string())


pivot_table = data_subset.groupby(['Latitude', 'Month'])['is_anomaly_multi'].mean().unstack()

# 2. Create the heatmap
plt.figure(figsize=(14, 8))

# Use imshow to create the heatmap. 'aspect=auto' adjusts the cell dimensions.
im = plt.imshow(pivot_table.values, 
                cmap='RdBu_r',  # Red-Blue colormap (reversed so red=high)
                aspect='auto', 
                extent=[1, 12, data_subset['Latitude'].max(), data_subset['Latitude'].min()])  # [xmin, xmax, ymax, ymin]

# Add a colorbar and label it
cbar = plt.colorbar(im, label='Anomaly Rate')
cbar.set_label('Anomaly Rate (Fraction of Measurements)', rotation=270, labelpad=20)

# Customize the axes and title
plt.xlabel('Month')
plt.ylabel('Latitude')
plt.title('Anomaly Rate by Latitude and Month', fontsize=14)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.yticks(np.arange(32, 54, 2))  # Latitude ticks from 32 to 54 in steps of 2

# Add a contour line to highlight the peak anomaly region
# This helps visualize patterns more clearly
X, Y = np.meshgrid(range(1, 13), pivot_table.index)
plt.contour(X, Y, pivot_table.values, levels=5, colors='black', linewidths=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig('figures/anomaly_heatmap_latitude_month.png', dpi=300, bbox_inches='tight')
plt.show()


import math

# Calculate number of rows and columns for the grid of plots
nmonths = 12
ncols = 4  # 4 columns
nrows = math.ceil(nmonths / ncols)  # 3 rows

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

for i, month in enumerate(range(1, 13)):
    # Get data for this month
    month_data = data_subset[data_subset['Month'] == month]
    
    # Create a 2D histogram of anomaly rate for this month
    # We'll use hexbin for better visualization of dense data
    hexbin = axes[i].hexbin(month_data['Longitude'], month_data['Latitude'], 
                            C=month_data['is_anomaly_multi'], 
                            reduce_C_function=np.mean,  # Show average anomaly rate in each hex
                            gridsize=30, 
                            cmap='RdBu_r')
    
    axes[i].set_title(f'Month: {month}')
    axes[i].set_xlabel('Longitude')
    axes[i].set_ylabel('Latitude')
    
    # Add colorbar for each subplot
    plt.colorbar(hexbin, ax=axes[i], label='Anomaly Rate')

# Hide any empty subplots
for j in range(nmonths, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('figures/anomaly_facet_grid_monthly_maps.png', dpi=150, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot a subset of points to avoid overplotting (otherwise too dense)
plot_sample = data_subset.sample(n=10000, random_state=42)  # Sample 10,000 points

# Create the 3D scatter plot. Color points by their anomaly flag.
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

# Adjust viewing angle for better perspective
ax.view_init(elev=20, azim=45)

plt.savefig('figures/3d_anomaly_distribution.png', dpi=150, bbox_inches='tight')
plt.show()