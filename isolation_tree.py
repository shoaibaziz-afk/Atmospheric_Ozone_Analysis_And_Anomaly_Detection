from data_loading import df
import numpy as np
import random

def isolation_tree(ozone_ppbv, current_depth = 0, max_depth = 10):
    if len(ozone_ppbv) <= 1 or current_depth >= max_depth:
        return {'depth': current_depth, 'samples': len(ozone_ppbv)}
    
    min_ozone_ppbv = min(ozone_ppbv)
    max_ozone_ppbv = max(ozone_ppbv)
    if min_ozone_ppbv == max_ozone_ppbv:
        return {'depth': current_depth, 'samples': len(ozone_ppbv)}
    
    split_value = random.uniform(min_ozone_ppbv, max_ozone_ppbv)
    left_values = [x for x in ozone_ppbv if x < split_value]
    right_values = [x for x in ozone_ppbv if x >= split_value]
    return {
        'split_value': split_value,
        'depth': current_depth,
        'left': isolation_tree(left_values, current_depth + 1, max_depth),
        'right': isolation_tree(right_values, current_depth + 1, max_depth)
    }
    
def find_path_length(tree, value, current_length = 0):
    if 'split_value' not in tree:
        return current_length + tree['depth']
    if value < tree['split_value']:
        return find_path_length(tree['left'], value, current_length + 1)
    else:
        return find_path_length(tree['right'], value, current_length + 1)
    
def run_isolation_tree(df):
    ozone_values = df['Ozone_ppbv'].tolist()
    num_trees = 20
    forest = []
    for i in range (num_trees):
        sample = random.sample(ozone_values, min(1000, len(ozone_values)))
        tree = isolation_tree(sample)
        forest.append(tree)
        
    anomaly_scores = []
    for value in ozone_values:
        total_path_length = 0
        for tree in forest:
            total_path_length += find_path_length(tree, value)
        avg_path_length = total_path_length / num_trees
        anomaly_score = 1.0 / (avg_path_length + 1)
        anomaly_scores.append(anomaly_score)
    
    # Add to dfframe
    df['simple_if_anomaly_score'] = anomaly_scores
    df['simple_if_outlier'] = df['simple_if_anomaly_score'] > np.percentile(anomaly_scores, 95)
    
    # Results
    outlier_count = df['simple_if_outlier'].sum()
    print(f"Found {outlier_count} outliers ({outlier_count/len(df):.1%})")
    
    # Save results
    output_path = 'results/simple_isolation_forest_results.csv'
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    return df