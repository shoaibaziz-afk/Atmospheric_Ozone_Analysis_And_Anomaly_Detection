from data_loading import df
import pandas as pd
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