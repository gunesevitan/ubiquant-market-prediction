from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

import settings
import metrics
import visualization


class XGBoostTrainer:

    def __init__(self, features, target, model_parameters, fit_parameters, seeds, model_directory):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.seeds = seeds
        self.model_directory = model_directory

    def train_and_validate_single_split(self, df):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features, target and folds
        """

        print(f'{"-" * 30}\nRunning XGBoost Model for Training\n{"-" * 30}\n')

        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)

        trn_idx, val_idx = df.loc[df['fold'] == 0].index, df.loc[df['fold'] == 1].index
        trn_dataset = xgb.DMatrix(df.loc[trn_idx, self.features], label=df.loc[trn_idx, self.target])
        val_dataset = xgb.DMatrix(df.loc[val_idx, self.features], label=df.loc[val_idx, self.target])
        seed_avg_val_predictions = np.zeros_like(df.loc[val_idx, self.target])
        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])

        for seed in self.seeds:

            print(f'\nSeed {seed}\n')
            self.model_parameters['seed'] = seed

            model = xgb.train(
                params=self.model_parameters,
                dtrain=trn_dataset,
                evals=[(trn_dataset, 'train'), (val_dataset, 'val')],
                num_boost_round=self.fit_parameters['boosting_rounds'],
                early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
                verbose_eval=self.fit_parameters['verbose_eval'],
                feval=metrics.pearson_correlation_coefficient_eval_xgb
            )
            # Save trained model
            model.save_model(settings.MODELS / self.model_directory / 'single_split' / f'model_seed{seed}.json')

            # Save validation predictions and feature importance scaled by the number of seeds
            val_predictions = model.predict(xgb.DMatrix(df.loc[val_idx, self.features]))
            seed_avg_val_predictions += (val_predictions / len(self.seeds))
            df.loc[val_idx, 'predictions'] = val_predictions
            for feature, importance in model.get_score(importance_type='gain').items():
                df_feature_importance.loc[feature, 'Importance'] += (importance / len(self.seeds))

            # Calculate mean pearson correlation coefficient on validation set
            val_score = metrics.mean_pearson_correlation_coefficient(df)
            print(f'\nXGBoost Validation Score: {val_score:.6f} (Seed: {seed})\n')

        df.loc[val_idx, 'predictions'] = seed_avg_val_predictions
        val_score = metrics.mean_pearson_correlation_coefficient(df.loc[val_idx, :])
        print(f'\nXGBoost Validation Score: {val_score:.6f}\n ({len(self.seeds)} Seed Average)')
        df['predictions'].to_csv(settings.MODELS / self.model_directory / 'single_split' / 'predictions.csv', index=False)

        # Visualize feature importance and predictions
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='XGBoost - Single Split Feature Importance (Gain)',
            path=settings.MODELS / self.model_directory / 'single_split' / 'feature_importance.png'
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

        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)

        trn_dataset = xgb.DMatrix(df.loc[:, self.features], label=df.loc[:, self.target])
        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])

        for seed in self.seeds:

            print(f'\nSeed {seed}\n')
            self.model_parameters['seed'] = seed

            model = xgb.train(
                params=self.model_parameters,
                dtrain=trn_dataset,
                evals=[(trn_dataset, 'train')],
                num_boost_round=self.fit_parameters['boosting_rounds'],
                early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
                verbose_eval=self.fit_parameters['verbose_eval'],
                feval=metrics.pearson_correlation_coefficient_eval_xgb
            )
            # Save trained model
            model.save_model(settings.MODELS / self.model_directory / 'no_split' / f'model_seed{seed}.json')

            # Save feature importance scaled by the number of seeds
            for feature, importance in model.get_score(importance_type='gain').items():
                df_feature_importance.loc[feature, 'Importance'] += (importance / len(self.seeds))

        # Visualize feature importance
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='XGBoost - No Split Feature Importance (Gain)',
            path=settings.MODELS / self.model_directory / 'no_split' / 'feature_importance.png'
        )
