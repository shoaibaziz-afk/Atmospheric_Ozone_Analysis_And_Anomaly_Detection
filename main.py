import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import math
from mpl_toolkits.mplot3d import Axes3D

# Import your custom functions (assuming they're in separate files)
from data_loading import load_data
from explore_ozone import train_linear_model, analyze_residuals, plot_residual_histogram, plot_model_fit # Linear Model
from find_anomalies import calculate_anomaly_thresholds, detect_anomalies, save_anomalies_to_csv, plot_anomalies # Anomalies Detection
from plot_more_anomalies import get_top_anomalies, plot_top_anomalies_context, calculate_annual_anomaly_rate, plot_annual_anomaly_rate # Top anomalies
from create_cyclic_encoding import prepare_features, train_multi_factor_model, plot_geographic_anomalies, plot_anomaly_rate_heatmap
from create_cyclic_encoding import plot_monthly_anomaly_rate, show_top_anomalies, plot_latitude_month_heatmap
from create_cyclic_encoding import plot_monthly_facet_grid, plot_3d_anomaly_distribution

def setup_directories():
    """Create necessary directories for output files"""
    import os
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("✅ Directories setup complete")

def run_linear_regression_analysis():
    """Run the linear regression analysis pipeline"""
    print("\n" + "="*60)
    print("LINEAR REGRESSION ANALYSIS")
    print("="*60)
    
    # Load and preprocess data
    filepath = 'Receptor_western_NAmerica_ozone_obs_1994_2021_from900to300.csv'
    df = load_data(filepath)
    
    # Prepare features and target
    X = df[['Pressure']]
    y = df['Ozone_ppbv']
    
    # Train the model
    model = train_linear_model(X, y)
    
    # Analyze residuals
    data_with_predictions = analyze_residuals(df, model)
    
    # Generate plots
    plot_residual_histogram(data_with_predictions)
    plot_model_fit(data_with_predictions, model)
    
    return data_with_predictions, model

def run_anomaly_detection_analysis(data):
    """Run anomaly detection pipeline"""
    print("\n" + "="*60)
    print("ANOMALY DETECTION ANALYSIS")
    print("="*60)
    
    # Calculate anomaly thresholds
    lower_bound, upper_bound, Q1, Q3, IQR = calculate_anomaly_thresholds(data)
    
    # Detect anomalies
    data_with_anomalies, anomaly_count = detect_anomalies(data, lower_bound, upper_bound)
    
    # Save results
    csv_path = save_anomalies_to_csv(data_with_anomalies)
    
    # Plot anomalies
    plot_anomalies(data_with_anomalies)
    
    return data_with_anomalies, lower_bound, upper_bound, anomaly_count

def run_top_anomalies_analysis(data, top_n=200):
    """Run top anomalies and annual analysis"""
    print("\n" + "="*60)
    print("TOP ANOMALIES & ANNUAL ANALYSIS")
    print("="*60)
    
    # Get top anomalies
    top_anomalies = get_top_anomalies(data, n=top_n)
    
    # Plot top anomalies in context
    plot_top_anomalies_context(data, top_anomalies, n=top_n)
    
    # Calculate annual anomaly rates
    annual_rate, annual_anomalies, annual_total = calculate_annual_anomaly_rate(data)
    
    # Plot annual anomaly rates
    plot_annual_anomaly_rate(annual_rate)
    
    return top_anomalies, annual_rate, annual_anomalies, annual_total

def run_create_cyclic_encoding(data):
    """Run multi-factor analysis pipeline"""
    print("\n" + "="*60)
    print("MULTI-FACTOR ANALYSIS")
    print("="*60)
    
    # Prepare features
    data_prepared = prepare_features(data.copy())
    
    # Train multi-factor model
    data_with_multi_anomalies, multi_model = train_multi_factor_model(data_prepared)
    
    # Generate all multi-factor plots
    plot_geographic_anomalies(data_with_multi_anomalies)
    plot_anomaly_rate_heatmap(data_with_multi_anomalies)
    plot_monthly_anomaly_rate(data_with_multi_anomalies)
    show_top_anomalies(data_with_multi_anomalies, n=10)
    plot_latitude_month_heatmap(data_with_multi_anomalies)
    plot_monthly_facet_grid(data_with_multi_anomalies)
    plot_3d_anomaly_distribution(data_with_multi_anomalies)
    
    return data_with_multi_anomalies, multi_model

def generate_summary_report(data_linear, data_anomalies, data_multi, top_anoms, annual_rate):
    """Generate a summary report of the analysis"""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    # Basic statistics
    total_points = len(data_linear)
    linear_anomalies = data_anomalies['is_anomaly'].sum()
    linear_anomaly_rate = (linear_anomalies / total_points) * 100
    
    print(f"Total data points: {total_points:,}")
    print(f"Linear model anomalies: {linear_anomalies:,} ({linear_anomaly_rate:.2f}%)")
    
    if 'is_anomaly_multi' in data_multi.columns:
        multi_anomalies = data_multi['is_anomaly_multi'].sum()
        multi_anomaly_rate = (multi_anomalies / total_points) * 100
        print(f"Multi-factor anomalies: {multi_anomalies:,} ({multi_anomaly_rate:.2f}%)")
    
    # Annual statistics
    print(f"\nAnnual anomaly rate range: {annual_rate.min():.2f}% - {annual_rate.max():.2f}%")
    print(f"Mean annual anomaly rate: {annual_rate.mean():.2f}%")
    
    # Top anomalies summary
    print(f"\nTop {len(top_anoms)} anomalies residual range:")
    print(f"  Min: {top_anoms['residual'].abs().min():.2f}")
    print(f"  Max: {top_anoms['residual'].abs().max():.2f}")
    print(f"  Mean: {top_anoms['residual'].abs().mean():.2f}")

def main():
    """Main function to execute the entire analysis pipeline"""
    print("🚀 Starting Ozone Data Analysis Pipeline")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    try:
        # Run linear regression analysis
        data_linear, linear_model = run_linear_regression_analysis()
        
        # Run anomaly detection
        data_anomalies, lower_bound, upper_bound, anomaly_count = run_anomaly_detection_analysis(data_linear)
        
        # Run top anomalies analysis
        top_anoms, annual_rate, annual_anoms, annual_total = run_top_anomalies_analysis(data_anomalies)
        
        # Run multi-factor analysis
        data_multi, multi_model = run_create_cyclic_encoding(data_anomalies)
        
        # Generate summary report
        generate_summary_report(data_linear, data_anomalies, data_multi, top_anoms, annual_rate)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("Check the 'figures' directory for all generated plots")
        print("Check the 'results' directory for output files")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure all plots are closed
        plt.close('all')

if __name__ == "__main__":
    main()