import numpy as np
import pandas as pd
import lightgbm as lgb

import settings
import metrics
import visualization


class LightGBMRanker:

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

        print(f'{"-" * 30}\nRunning LightGBM Ranker Model for Training\n{"-" * 30}\n')

        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])
        trn_idx, val_idx = df.loc[df['fold'] == 0].index, df.loc[df['fold'] == 1].index
        X_trn, y_trn, query_trn = df.loc[trn_idx, self.features], df.loc[trn_idx, self.target], [len(trn_idx)]
        X_val, y_val, query_val = df.loc[val_idx, self.features], df.loc[val_idx, self.target], [len(val_idx)]

        model = lgb.LGBMRanker(**self.model_parameters)
        model.fit(
            X_trn, y_trn, group=query_trn,
            eval_set=[(X_val, y_val)], eval_group=[query_val],
            eval_at=self.fit_parameters['eval_at'],
            eval_metric=[metrics.pearson_correlation_coefficient_eval_lgb],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            categorical_feature=self.categorical_features
        )
        model.booster_.save_model(
            settings.MODELS / 'lightgbm_ranker' / 'single_split' / 'model',
            num_iteration=None,
            start_iteration=0,
            importance_type='gain'
        )

        df.loc[val_idx, 'predictions'] = model.predict(df.loc[val_idx, self.features])
        df_feature_importance['Importance'] += model.feature_importances_(importance_type='gain')
        val_score = metrics.mean_pearson_correlation_coefficient(df)
        print(f'\nLightGBM Ranker Validation Score: {val_score:.6f}\n')
        df['predictions'].to_csv(settings.MODELS / 'lightgbm_ranker' / 'single_split' / 'predictions.csv', index=False)

        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='LightGBM Ranker - Single Split Feature Importance (Gain)',
            path=settings.MODELS / 'lightgbm_ranker' / 'single_split' / 'feature_importance.png'
        )
        visualization.visualize_predictions(
            y_true=df.loc[val_idx, self.target],
            y_pred=df.loc[val_idx, 'predictions'],
            title='LightGBM Ranker - Single Split Predictions',
            path=settings.MODELS / 'lightgbm_ranker' / 'single_split' / 'predictions.png'
        )

    def train_no_split(self, df):

        """
        Train on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features and target
        """

        print(f'{"-" * 30}\nRunning LightGBM Ranker Model for Training\n{"-" * 30}\n')

        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])
        X_trn, y_trn, query_trn = df.loc[:, self.features], df.loc[:, self.target], [len(df)]

        model = lgb.LGBMRanker(**self.model_parameters)
        model.fit(
            X_trn, y_trn, group=query_trn,
            eval_set=None, eval_group=None,
            eval_at=self.fit_parameters['eval_at'],
            eval_metric=[metrics.pearson_correlation_coefficient_eval_lgb],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            categorical_feature=self.categorical_features
        )
        model.booster_.save_model(
            settings.MODELS / 'lightgbm_ranker' / 'no_split' / 'model',
            num_iteration=None,
            start_iteration=0,
            importance_type='gain'
        )

        df_feature_importance['Importance'] += model.feature_importances_(importance_type='gain')
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='LightGBM Ranker - Single Split Feature Importance (Gain)',
            path=settings.MODELS / 'lightgbm_ranker' / 'no_split' / 'feature_importance.png'
        )
