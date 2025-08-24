from explore_ozone import data_subset
import matplotlib.pyplot as plt

Q1 = data_subset['residual'].quantile(0.25)
Q3 = data_subset['residual'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n[ANOMALY THRESHOLD CALCULATION]")
print(f"Q1 (25th percentile): {Q1:.2f}")
print(f"Q3 (75th percentile): {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower bound (anomaly if residual < {lower_bound:.2f})")
print(f"Upper bound (anomaly if residual > {upper_bound:.2f})")

data_subset['is_anomaly'] = (data_subset['residual'] < lower_bound) | (data_subset['residual'] > upper_bound)
anomaly_count = data_subset['is_anomaly'].sum()
total_count = len(data_subset)
print(f"\n[ANOMALY COUNT] {anomaly_count} anomalies found out of {total_count} total points ({anomaly_count/total_count:.2%})")

plt.figure(figsize=(10, 6))

# Plot all normal points in grey
plt.scatter(data_subset[~data_subset['is_anomaly']]['Ozone_ppbv'],
            data_subset[~data_subset['is_anomaly']]['Pressure'],
            alpha=0.1, s=5, color='grey', label='Normal')

# Plot all anomaly points in red
plt.scatter(data_subset[data_subset['is_anomaly']]['Ozone_ppbv'],
            data_subset[data_subset['is_anomaly']]['Pressure'],
            alpha=0.7, s=20, color='red', label='Anomaly')

plt.gca().invert_yaxis()
plt.xlabel('Ozone (ppbv)')
plt.ylabel('Pressure (hPa)')
plt.title('Anomalies in Tropospheric Ozone Data (IQR Method)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/anomalies_detected.png')
plt.show()