import argparse
import yaml
import pandas as pd

import settings
from linear_model_trainer import LinearModelTrainer
from lgb_trainer import LightGBMTrainer
from xgb_trainer import XGBoostTrainer
from ae_trainer import AutoEncoderTrainer
from nn_trainer import NeuralNetworkTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    df = pd.read_pickle(settings.DATA / 'train.pkl')

    if config['validation_type'] == 'single_split':
        df = df.merge(pd.read_pickle(settings.DATA / 'folds.pkl'), on=['time_id', 'investment_id'], how='left')

    if config['model'] == 'linear_model':

        trainer = LinearModelTrainer(
            features=config['features'],
            target=config['target'],
            model_class=config['model_class'],
            model_parameters=config['model_parameters']
        )

    elif config['model'] == 'lightgbm':

        trainer = LightGBMTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            fit_parameters=config['fit_parameters'],
            categorical_features=config['categorical_features'],
            seeds=config['seeds'],
            model_directory=config['model_directory']
        )

    elif config['model'] == 'xgboost':

        trainer = XGBoostTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            fit_parameters=config['fit_parameters'],
            seeds=config['seeds'],
            model_directory=config['model_directory']
        )

    elif config['model'] == 'auto_encoder':

        trainer = AutoEncoderTrainer(
            features=config['features'],
            model_parameters=config['model_parameters'],
            training_parameters=config['training_parameters'],
        )

    elif config['model'] == 'neural_network':

        trainer = NeuralNetworkTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            training_parameters=config['training_parameters'],
            seeds=config['seeds'],
            model_directory=config['model_directory']
        )

    if args.mode == 'train':
        if config['validation_type'] == 'single_split':
            trainer.train_and_validate_single_split(df)
        elif config['validation_type'] == 'no_split':
            trainer.train_no_split(df)
