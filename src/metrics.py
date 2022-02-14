import numpy as np
from scipy.stats import pearsonr


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


def pearson_correlation_coefficient_eval_lgb(y_pred, train_dataset):

    """
    Calculate Pearson correlation coefficient between ground-truth and predictions

    Parameters
    ----------
    y_pred [array-like of shape (n_samples)]: Predictions
    train_dataset (lightgbm.Dataset): Training dataset

    Returns
    -------
    eval_name (str): Name of the evaluation metric
    eval_result (float): Result of the evaluation metric
    is_higher_better (bool): Whether the higher is better or worse for the evaluation metric
    """

    eval_name = 'pearson\'s r'
    y_true = train_dataset.get_label()
    eval_result = pearsonr(y_true, y_pred)[0]
    is_higher_better = True

    return eval_name, eval_result, is_higher_better


def pearson_correlation_coefficient_eval_xgb(y_pred, train_dataset):

    """
    Calculate Pearson correlation coefficient between ground-truth and predictions

    Parameters
    ----------
    y_pred [array-like of shape (n_samples)]: Predictions
    train_dataset (xgboost.DMatrix): Training dataset

    Returns
    -------
    eval_name (str): Name of the evaluation metric
    eval_result (float): Result of the evaluation metric
    """

    eval_name = 'pearson\'s r'
    y_true = train_dataset.get_label()
    eval_result = float(pearsonr(y_true, y_pred)[0])

    return eval_name, eval_result
