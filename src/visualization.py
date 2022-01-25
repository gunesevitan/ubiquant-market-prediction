import matplotlib.pyplot as plt
import seaborn as sns


def visualize_feature_importance(df_feature_importance, title, path=None):

    """
    Visualize feature importance of the models

    Parameters
    ----------
    df_feature_importance [pandas.DataFrame of shape (n_features)]: DataFrame of features as index and importance as values
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    df_feature_importance.sort_values(by='Importance', inplace=True, ascending=False)

    fig, ax = plt.subplots(figsize=(24, len(df_feature_importance)))
    sns.barplot(
        x='Importance',
        y=df_feature_importance.index,
        data=df_feature_importance,
        palette='Blues_d',
        ax=ax
    )
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_predictions(y_true, y_pred, title, path=None):

    """
    Visualize predictions of the models

    Parameters
    ----------
    y_true [array-like of shape (n_samples)]: Ground-truth
    y_pred [array-like of shape (n_samples)]: Predictions
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, axes = plt.subplots(ncols=2, figsize=(32, 8))
    fig.subplots_adjust(top=0.85)
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0])
    sns.histplot(y_true, label='Ground-truth Labels', kde=True, color='blue', ax=axes[1])
    sns.histplot(y_pred, label='Predictions', kde=True, color='red', ax=axes[1])
    axes[0].set_xlabel(f'Ground-truth Labels', size=15, labelpad=12.5)
    axes[0].set_ylabel(f'Predictions', size=15, labelpad=12.5)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].legend(prop={'size': 17.5})
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5, pad=10)
        axes[i].tick_params(axis='y', labelsize=12.5, pad=10)
    axes[0].set_title(f'Ground-truth Labels vs Predictions', size=20, pad=15)
    axes[1].set_title(f'Predictions Distributions', size=20, pad=15)

    fig.suptitle(title, size=20, y=0.95)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
