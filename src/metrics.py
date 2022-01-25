import numpy as np


def _pearson_correlation_coefficient(df):

    """
    Calculate Pearson correlation coefficient between target and predictions on given dataframe

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 2)]: Training set with target and predictions columns
    """

    return df.corr()['target']['predictions']


def mean_pearson_correlation_coefficient(df):

    """
    Calculate Pearson correlation coefficient between target and predictions for every time_id average them on given dataframe

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 3)]: Training set with time_id, target and predictions columns
    """

    return np.mean(df[['time_id', 'target', 'predictions']].groupby('time_id').apply(_pearson_correlation_coefficient))
