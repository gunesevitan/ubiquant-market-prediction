import sys
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


DTYPES = {f'f_{i}': np.float32 for i in range(300)}
DTYPES['investment_id'] = np.uint16
DTYPES['time_id'] = np.uint16
DTYPES['target'] = np.float32


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train.csv', usecols=list(DTYPES.keys()), dtype=DTYPES)
    print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Write train.csv as a pickle file that consumes less space and loads faster
    # Pickle file consumes less space since it is a serialized python object
    # 18.5 GB csv file is compressed to 3.8 GB pickle file
    df_train.to_pickle(settings.DATA / 'train.pkl')
