import pandas as pd

filepath = 'Receptor_western_NAmerica_ozone_obs_1994_2021_from900to300.csv'

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

df = load_data(filepath)
print(df)