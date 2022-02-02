import numpy as np
import pandas as pd
import lightgbm as lgb

import settings
import metrics
import visualization


class LightGBMTrainer:

    def __init__(self, features, target, model_parameters, fit_parameters, categorical_features):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.categorical_features = categorical_features

    def train_and_validate_single_split(self, df):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features, target and folds
        """

        print(f'{"-" * 30}\nRunning LightGBM Model for Training\n{"-" * 30}\n')

        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])
        trn_idx, val_idx = df.loc[df['fold'] == 0].index, df.loc[df['fold'] == 1].index
        trn_dataset = lgb.Dataset(df.loc[trn_idx, self.features], label=df.loc[trn_idx, self.target], categorical_feature=self.categorical_features)
        val_dataset = lgb.Dataset(df.loc[val_idx, self.features], label=df.loc[val_idx, self.target], categorical_feature=self.categorical_features)

        model = lgb.train(
            params=self.model_parameters,
            train_set=trn_dataset,
            valid_sets=[trn_dataset, val_dataset],
            num_boost_round=self.fit_parameters['boosting_rounds'],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            verbose_eval=self.fit_parameters['verbose_eval'],
            feval=[metrics.pearson_correlation_coefficient_eval_lgb]
        )
        model.save_model(
            settings.MODELS / 'lightgbm' / 'single_split' / 'model',
            num_iteration=None,
            start_iteration=0,
            importance_type='gain'
        )

        df.loc[val_idx, 'predictions'] = model.predict(df.loc[val_idx, self.features])
        df_feature_importance['Importance'] += model.feature_importance(importance_type='gain')
        val_score = metrics.mean_pearson_correlation_coefficient(df)
        print(f'\nLightGBM Validation Score: {val_score:.6f}\n')
        df['predictions'].to_csv(settings.MODELS / 'lightgbm' / 'single_split' / 'predictions.csv', index=False)

        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='LightGBM - Single Split Feature Importance (Gain)',
            path=settings.MODELS / 'lightgbm' / 'single_split' / 'feature_importance.png'
        )
        visualization.visualize_predictions(
            y_true=df.loc[val_idx, self.target],
            y_pred=df.loc[val_idx, 'predictions'],
            title='LightGBM - Single Split Predictions',
            path=settings.MODELS / 'lightgbm' / 'single_split' / 'predictions.png'
        )

    def train_no_split(self, df):

        """
        Train on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features and target
        """

        print(f'{"-" * 30}\nRunning LightGBM Model for Training\n{"-" * 30}\n')

        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])
        trn_dataset = lgb.Dataset(df.loc[:, self.features], label=df.loc[:, self.target], categorical_feature=self.categorical_features)

        model = lgb.train(
            params=self.model_parameters,
            train_set=trn_dataset,
            valid_sets=[trn_dataset],
            num_boost_round=self.fit_parameters['boosting_rounds'],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            verbose_eval=self.fit_parameters['verbose_eval'],
            feval=[metrics.pearson_correlation_coefficient_eval_lgb]
        )
        model.save_model(
            settings.MODELS / 'lightgbm' / 'no_split' / 'model',
            num_iteration=None,
            start_iteration=0,
            importance_type='gain'
        )

        df_feature_importance['Importance'] += model.feature_importance(importance_type='gain')
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='LightGBM - No Split Feature Importance (Gain)',
            path=settings.MODELS / 'lightgbm' / 'no_split' / 'feature_importance.png'
        )
