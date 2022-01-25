import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.append('..')
import settings


def get_group_folds(df, n_splits, verbose=False):

    """
    Create a column of folds with specified configuration of group k-fold on given training set

    Parameters
    ----------
    df [pandas.DataFrame of shape (3141410, 303)]: Training set
    n_splits (int): Number of folds (2 <= n_splits)
    verbose (bool): Flag for verbosity
    """

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(gkf.split(X=df, groups=df['time_id']), 1):
        df.loc[val_idx, 'fold'] = fold
    df['fold'] = df['fold'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set split into {n_splits} group folds')
        for fold in range(1, n_splits + 1):
            df_fold = df[df['fold'] == fold]
            print(f'Fold {fold} {df_fold.shape} - target mean: {df_fold["target"].mean():.4} std: {df_fold["target"].std():.4} min: {df_fold["target"].min():.4} max: {df_fold["target"].max():.4}')


def get_time_id_splits(df, validation_range, verbose=False):

    """
    Create a column of splits with specified validation range on given training set

    Parameters
    ----------
    df [pandas.DataFrame of shape (3141410, 303)]: Training set
    validation_range (tuple): Start and end time_id of validation split
    verbose (bool): Flag for verbosity
    """

    df['fold'] = 0
    val_idx = (df['time_id'] >= validation_range[0]) & (df['time_id'] <= validation_range[1])
    df.loc[val_idx, 'fold'] = 1
    df['fold'] = df['fold'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set split into training and validation sets')
        for fold in range(2):
            df_fold = df[df['fold'] == fold]
            print(f'Fold {fold} {df_fold.shape} - target mean: {df_fold["target"].mean():.4} std: {df_fold["target"].std():.4} min: {df_fold["target"].min():.4} max: {df_fold["target"].max():.4}')


if __name__ == '__main__':

    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    # Using samples between time_id 1026 and 1219 as the validation set
    # Number of samples in that range correspond to 20% of the entire training set
    get_time_id_splits(df_train, validation_range=(1026, 1219), verbose=True)
    df_train[['time_id', 'investment_id', 'fold']].to_pickle(settings.DATA / 'folds.pkl')
