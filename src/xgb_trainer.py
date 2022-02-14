import numpy as np
import pandas as pd
import xgboost as xgb

import settings
import metrics
import visualization


class XGBoostTrainer:

    def __init__(self, features, target, model_parameters, fit_parameters):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters

    def train_and_validate_single_split(self, df):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features, target and folds
        """

        print(f'{"-" * 30}\nRunning XGBoost Model for Training\n{"-" * 30}\n')

        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])
        trn_idx, val_idx = df.loc[df['fold'] == 0].index, df.loc[df['fold'] == 1].index
        trn_dataset = xgb.DMatrix(df.loc[trn_idx, self.features], label=df.loc[trn_idx, self.target])
        val_dataset = xgb.DMatrix(df.loc[val_idx, self.features], label=df.loc[val_idx, self.target])

        model = xgb.train(
            params=self.model_parameters,
            dtrain=trn_dataset,
            evals=[(trn_dataset, 'train'), (val_dataset, 'val')],
            num_boost_round=self.fit_parameters['boosting_rounds'],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            verbose_eval=self.fit_parameters['verbose_eval'],
            feval=metrics.pearson_correlation_coefficient_eval_xgb
        )
        model.save_model(settings.MODELS / 'xgboost' / 'single_split' / 'model.json')

        df.loc[val_idx, 'predictions'] = model.predict(xgb.DMatrix(df.loc[val_idx, self.features]))
        for feature, importance in model.get_score(importance_type='gain').items():
            df_feature_importance.loc[feature, 'Importance'] += importance

        val_score = metrics.mean_pearson_correlation_coefficient(df)
        print(f'\nXGBoost Validation Score: {val_score:.6f}\n')
        df['predictions'].to_csv(settings.MODELS / 'xgboost' / 'single_split' / 'predictions.csv', index=False)

        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='XGBoost - Single Split Feature Importance (Gain)',
            path=settings.MODELS / 'xgboost' / 'single_split' / 'feature_importance.png'
        )
        visualization.visualize_predictions(
            y_true=df.loc[val_idx, self.target],
            y_pred=df.loc[val_idx, 'predictions'],
            title='XGBoost - Single Split Predictions',
            path=settings.MODELS / 'xgboost' / 'single_split' / 'predictions.png'
        )

    def train_no_split(self, df):

        """
        Train on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features and target
        """

        print(f'{"-" * 30}\nRunning XGBoost Model for Training\n{"-" * 30}\n')

        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])
        trn_dataset = xgb.DMatrix(df.loc[:, self.features], label=df.loc[:, self.target])

        model = xgb.train(
            params=self.model_parameters,
            dtrain=trn_dataset,
            evals=[(trn_dataset, 'train')],
            num_boost_round=self.fit_parameters['boosting_rounds'],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            verbose_eval=self.fit_parameters['verbose_eval'],
            feval=metrics.pearson_correlation_coefficient_eval_xgb
        )
        model.save_model(settings.MODELS / 'xgboost' / 'no_split' / 'model.json')

        for feature, importance in model.get_score(importance_type='gain').items():
            df_feature_importance.loc[feature, 'Importance'] += importance
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='XGBoost - No Split Feature Importance (Gain)',
            path=settings.MODELS / 'xgboost' / 'no_split' / 'feature_importance.png'
        )
