from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import sklearn.linear_model

import settings
import metrics
import visualization


class LinearModelTrainer:

    def __init__(self, features, target, model_class, model_parameters, model_directory):

        self.features = features
        self.target = target
        self.model_class = model_class
        self.model_parameters = model_parameters
        self.model_directory = model_directory

    def train_and_validate_single_split(self, df):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features, target and folds
        """

        print(f'{"-" * 30}\nRunning {self.model_class} Model for Training\n{"-" * 30}\n')

        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)
        trn_idx, val_idx = df.loc[df['fold'] == 0].index, df.loc[df['fold'] == 1].index
        df_feature_coefficient = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])

        model = getattr(sklearn.linear_model, self.model_class)(**self.model_parameters)
        model.fit(df.loc[trn_idx, self.features], df.loc[trn_idx, self.target])
        # Save trained model
        pickle.dump(model, open(settings.MODELS / self.model_directory / 'single_split' / 'model.pkl', 'wb'))

        # Save validation predictions and feature coefficients
        val_predictions = model.predict(df.loc[val_idx, self.features])
        df.loc[val_idx, 'predictions'] = val_predictions
        df_feature_coefficient['Importance'] += model.coef_

        # Calculate mean pearson correlation coefficient on validation set
        val_score = metrics.mean_pearson_correlation_coefficient(df.loc[val_idx, :])
        print(f'\n{self.model_class} Validation Score: {val_score:.6f}\n')
        df['predictions'].to_csv(settings.MODELS / self.model_directory / 'single_split' / 'predictions.csv', index=False)

        # Visualize feature importance and predictions
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_coefficient,
            title=f'{self.model_class} - Single Split Feature Coefficient',
            path=settings.MODELS / self.model_directory / 'single_split' / 'feature_coefficient.png'
        )
        visualization.visualize_predictions(
            y_true=df.loc[val_idx, self.target],
            y_pred=df.loc[val_idx, 'predictions'],
            title=f'{self.model_class} - Single Split Predictions',
            path=settings.MODELS / self.model_directory / 'single_split' / 'predictions.png'
        )

    def train_no_split(self, df):

        """
        Train on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features and target
        """

        print(f'{"-" * 30}\nRunning {self.model_class} Model for Training\n{"-" * 30}\n')
        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)
        df_feature_coefficient = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])

        model = getattr(sklearn.linear_model, self.model_class)(**self.model_parameters)
        model.fit(df.loc[:, self.features], df.loc[:, self.target])
        # Save trained model
        pickle.dump(model, open(settings.MODELS / self.model_directory / 'no_split' / 'model', 'wb'))

        # Save feature coefficients
        df_feature_coefficient['Importance'] += model.coef_
        # Visualize feature importance
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_coefficient,
            title=f'{self.model_class} - No Split Feature Coefficient',
            path=settings.MODELS / 'linear_model' / 'no_split' / 'feature_coefficient.png'
        )
