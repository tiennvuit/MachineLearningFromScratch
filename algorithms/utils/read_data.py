import pandas as pd
import numpy as np
import tqdm


def read_data_from_csv(path):
    data = pd.read_csv(path)
    X = []
    y = []
    for _, row in data.iterrows():
        X.append((row['longitude'], row['latitude'], 
                row['housing_median_age'], row['total_rooms'],
                row['total_bedrooms'], row['population'],
                row['households'], row['median_income'],
                #row['ocean_proximity']
        ))
        y.append(row['median_income'])

    return np.array(X), np.array(y)


def read_data(path: str):
    if path.endswith('.csv'):
        data = read_data_from_csv(path=path)
    else:
        pass
    return data