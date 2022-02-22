from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

import settings
import metrics
import visualization


class LightGBMTrainer:

    def __init__(self, features, target, model_parameters, fit_parameters, categorical_features, seeds, model_directory):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.categorical_features = categorical_features
        self.seeds = seeds
        self.model_directory = model_directory

    def train_and_validate_single_split(self, df):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features, target and folds
        """

        print(f'{"-" * 30}\nRunning LightGBM Model for Training\n{"-" * 30}\n')

        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)
        trn_idx, val_idx = df.loc[df['fold'] == 0].index, df.loc[df['fold'] == 1].index
        seed_avg_val_predictions = np.zeros_like(df.loc[val_idx, self.target])
        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])

        for seed in self.seeds:

            print(f'\nSeed {seed}\n')
            self.model_parameters['seed'] = seed
            self.model_parameters['feature_fraction_seed'] = seed
            self.model_parameters['bagging_seed'] = seed
            self.model_parameters['drop_seed'] = seed
            self.model_parameters['data_random_seed'] = seed

            trn_dataset = lgb.Dataset(df.loc[trn_idx, self.features], label=df.loc[trn_idx, self.target], categorical_feature=self.categorical_features)
            val_dataset = lgb.Dataset(df.loc[val_idx, self.features], label=df.loc[val_idx, self.target], categorical_feature=self.categorical_features)

            model = lgb.train(
                params=self.model_parameters,
                train_set=trn_dataset,
                valid_sets=[trn_dataset, val_dataset],
                num_boost_round=self.fit_parameters['boosting_rounds'],
                callbacks=[
                    lgb.early_stopping(self.fit_parameters['early_stopping_rounds']),
                    lgb.log_evaluation(self.fit_parameters['verbose_eval'])
                ],
                feval=[metrics.pearson_correlation_coefficient_eval_lgb]
            )
            # Save trained model
            model.save_model(
                settings.MODELS / self.model_directory / 'single_split' / f'model_seed{seed}.txt',
                num_iteration=None,
                start_iteration=0,
                importance_type='gain'
            )

            # Save validation predictions and feature importance scaled by the number of seeds
            val_predictions = model.predict(df.loc[val_idx, self.features])
            seed_avg_val_predictions += (val_predictions / len(self.seeds))
            df.loc[val_idx, 'predictions'] = val_predictions
            df_feature_importance['Importance'] += (model.feature_importance(importance_type='gain') / len(self.seeds))

            # Calculate mean pearson correlation coefficient on validation set
            val_score = metrics.mean_pearson_correlation_coefficient(df.loc[val_idx, :])
            print(f'\nLightGBM Validation Score: {val_score:.6f} (Seed: {seed})\n')

        df.loc[val_idx, 'predictions'] = seed_avg_val_predictions
        val_score = metrics.mean_pearson_correlation_coefficient(df.loc[val_idx, :])
        print(f'\nLightGBM Validation Score: {val_score:.6f}\n ({len(self.seeds)} Seed Average)')
        df['predictions'].to_csv(settings.MODELS / self.model_directory / 'single_split' / 'predictions.csv', index=False)

        # Visualize feature importance and predictions
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='LightGBM - Single Split Feature Importance (Gain)',
            path=settings.MODELS / self.model_directory / 'single_split' / 'feature_importance.png'
        )
        visualization.visualize_predictions(
            y_true=df.loc[val_idx, self.target],
            y_pred=df.loc[val_idx, 'predictions'],
            title='LightGBM - Single Split Predictions',
            path=settings.MODELS / self.model_directory / 'single_split' / 'predictions.png'
        )

    def train_no_split(self, df):

        """
        Train on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features and target
        """

        print(f'{"-" * 30}\nRunning LightGBM Model for Training\n{"-" * 30}\n')

        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)
        df_feature_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])

        for seed in self.seeds:

            print(f'\nSeed {seed}\n')
            self.model_parameters['seed'] = seed
            self.model_parameters['feature_fraction_seed'] = seed
            self.model_parameters['bagging_seed'] = seed
            self.model_parameters['drop_seed'] = seed
            self.model_parameters['data_random_seed'] = seed

            trn_dataset = lgb.Dataset(df.loc[:, self.features], label=df.loc[:, self.target], categorical_feature=self.categorical_features)

            model = lgb.train(
                params=self.model_parameters,
                train_set=trn_dataset,
                valid_sets=[trn_dataset],
                num_boost_round=self.fit_parameters['boosting_rounds'],
                callbacks=[
                    lgb.early_stopping(self.fit_parameters['early_stopping_rounds']),
                    lgb.log_evaluation(self.fit_parameters['verbose_eval'])
                ],
                feval=[metrics.pearson_correlation_coefficient_eval_lgb]
            )
            # Save trained model
            model.save_model(
                settings.MODELS / self.model_directory / 'no_split' / f'model_seed{seed}.txt',
                num_iteration=None,
                start_iteration=0,
                importance_type='gain'
            )

            # Save feature importance scaled by the number of seeds
            df_feature_importance['Importance'] += (model.feature_importance(importance_type='gain') / len(self.seeds))

        # Visualize feature importance
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='LightGBM - No Split Feature Importance (Gain)',
            path=settings.MODELS / self.model_directory / 'no_split' / 'feature_importance.png'
        )
