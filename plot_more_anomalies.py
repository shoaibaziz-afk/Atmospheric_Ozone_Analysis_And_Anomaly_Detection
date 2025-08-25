from explore_ozone import data_subset
import matplotlib.pyplot as plt

top_anomalies = data_subset.sort_values(by='residual', key=abs, ascending=False).head(200)

print("\n[TOP 20 MOST EXTREME ANOMALIES]")
print(top_anomalies[['Pressure', 'Ozone_ppbv', 'predicted_ozone', 'residual']])

plt.figure(figsize=(10, 6))
# Plot all points
plt.scatter(data_subset['Ozone_ppbv'], data_subset['Pressure'], alpha=0.1, s=2, color='grey', label='Normal')
# Highlight the top 20 most extreme anomalies
plt.scatter(top_anomalies['Ozone_ppbv'], top_anomalies['Pressure'], alpha=1, s=50, color='red', label='Top 20 Anomalies')

plt.gca().invert_yaxis()
plt.xlabel('Ozone (ppbv)')
plt.ylabel('Pressure (hPa)')
plt.title('Top 20 Anomalies in Context')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/top_anomalies_context.png')
plt.show()

annual_anomalies = data_subset[data_subset['is_anomaly']].groupby('Year').size()
annual_total = data_subset.groupby('Year').size()

# Calculate anomaly rate per year
annual_rate = (annual_anomalies / annual_total) * 100  # Percentage

# Plot the annual anomaly rate
plt.figure(figsize=(12, 5))
annual_rate.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Anomaly Rate (%)')
plt.title('Percentage of Anomalous Measurements per Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/anomaly_rate_by_year.png')
plt.show()