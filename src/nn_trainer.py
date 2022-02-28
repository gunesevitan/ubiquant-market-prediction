from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import settings
import metrics
import visualization
import training_utils
from datasets import TabularDataset
import torch_modules


class NeuralNetworkTrainer:

    def __init__(self, features, target, model_parameters, training_parameters, seeds, model_directory):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.seeds = seeds
        self.model_directory = model_directory

    def train_fn(self, train_loader, model, criterion, optimizer, device, scheduler=None):

        """
        Train given model on given data loader

        Parameters
        ----------
        train_loader (torch.utils.data.DataLoader): Training set data loader
        model (torch.nn.Module): Model to train
        criterion (torch.nn.modules.loss): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Location of the model and inputs
        scheduler (torch.optim.LRScheduler or None): Learning rate scheduler

        Returns
        -------
        train_loss (float): Average training loss after model is fully trained on training set data loader
        """

        print('\n')
        model.train()
        progress_bar = tqdm(train_loader)
        losses = []

        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            average_loss = np.mean(losses)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {lr:.8f}')

        train_loss = np.mean(losses)
        return train_loss

    def val_fn(self, val_loader, model, criterion, device):

        """
        Validate given model on given data loader

        Parameters
        ----------
        val_loader (torch.utils.data.DataLoader): Validation set data loader
        model (torch.nn.Module): Model to validate
        criterion (torch.nn.modules.loss): Loss function
        device (torch.device): Location of the model and inputs

        Returns
        -------
        val_loss (float): Average validation loss after model is fully validated on validation set data loader
        val_pearson_r (float): Pearson correlation coefficient calculated on the validation set
        """

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []
        targets = []
        predictions = []

        with torch.no_grad():
            for features, labels in progress_bar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_loss: {average_loss:.6f}')
                targets += labels.detach().cpu().numpy().tolist()
                predictions += outputs.detach().cpu().numpy().tolist()

        val_loss = np.mean(losses)
        val_predictions = np.array(predictions)
        val_pearson_r = pearsonr(targets, predictions)[0]
        return val_loss, val_pearson_r, val_predictions

    def train_and_validate_single_split(self, df):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features, target and folds
        """

        print(f'\n{"-" * 30}\nRunning Neural Network Model for Training\n{"-" * 30}\n')
        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)
        trn_idx, val_idx = df.loc[df['fold'] == 0].index, df.loc[df['fold'] == 1].index
        seed_avg_val_predictions = np.zeros_like(df.loc[val_idx, self.target])

        train_dataset = TabularDataset(df.loc[trn_idx, self.features].values, df.loc[trn_idx, self.target].values)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_parameters['data_loader']['training_batch_size'],
            sampler=RandomSampler(train_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )
        val_dataset = TabularDataset(df.loc[val_idx, self.features].values, df.loc[val_idx, self.target].values)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_parameters['data_loader']['test_batch_size'],
            sampler=SequentialSampler(val_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )

        for seed in self.seeds:

            print(f'\nSeed {seed}\n')

            # Set model, loss function, device and seed for reproducible results
            training_utils.set_seed(seed, deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])
            criterion = getattr(torch_modules, self.training_parameters['loss_function'])(**self.training_parameters['loss_args'])
            model = getattr(torch_modules, self.model_parameters['model_class'])(**self.model_parameters['model_args'])
            if self.model_parameters['model_checkpoint_path'] is not None:
                model_checkpoint_path = self.model_parameters['model_checkpoint_path']
                model.load_state_dict(torch.load(model_checkpoint_path))
            model.to(device)

            # Set optimizer and learning rate scheduler
            optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_args']) if self.training_parameters['lr_scheduler'] is not None else None

            early_stopping = False
            summary = {
                'train_loss': [],
                'val_loss': []
            }

            for epoch in range(1, self.training_parameters['epochs'] + 1):

                if early_stopping:
                    break

                if self.training_parameters['lr_scheduler'] == 'ReduceLROnPlateau':
                    # Step on validation loss if learning rate scheduler is ReduceLROnPlateau
                    train_loss = self.train_fn(train_loader, model, criterion, optimizer, device, scheduler=None)
                    val_loss, val_pearson_r, val_predictions = self.val_fn(val_loader, model, criterion, device)
                    scheduler.step(val_loss)
                else:
                    # Learning rate scheduler will work in validation function if it is not ReduceLROnPlateau
                    train_loss = self.train_fn(train_loader, model, criterion, optimizer, device, scheduler)
                    val_loss, val_pearson_r, val_predictions = self.val_fn(val_loader, model, criterion, device)

                print(f'Epoch {epoch} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f} Pearson\'s R: {val_pearson_r:.6f}')
                best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                if val_loss < best_val_loss:
                    # Save model and validation set predictions if validation loss improves
                    model_path = settings.MODELS / self.model_directory / 'single_split' / f'model_seed{seed}.pt'
                    torch.save(model.state_dict(), model_path)
                    print(f'Saving model to {model_path} (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})')
                    df.loc[val_idx, 'predictions'] = val_predictions

                summary['train_loss'].append(train_loss)
                summary['val_loss'].append(val_loss)

                best_iteration = np.argmin(summary['val_loss']) + 1
                if len(summary['val_loss']) - best_iteration >= self.training_parameters['early_stopping_patience']:
                    print(f'Early stopping (validation loss didn\'t increase for {self.training_parameters["early_stopping_patience"]} epochs/steps)')
                    print(f'Best validation loss is {np.min(summary["val_loss"]):.6f}')
                    early_stopping = True

            # Save validation set predictions with latest best loss
            seed_avg_val_predictions += (df.loc[val_idx, 'predictions'] / len(self.seeds))

            # Calculate mean pearson correlation coefficient on validation set
            val_score = metrics.mean_pearson_correlation_coefficient(df.loc[val_idx, :])
            print(f'\nNeural Network Validation Score: {val_score:.6f} (Seed: {seed})\n')
            df['predictions'].to_csv(settings.MODELS / self.model_directory / 'single_split' / 'predictions.csv', index=False)

            # Visualize learning curve and predictions
            visualization.visualize_learning_curve(
                training_losses=summary['train_loss'],
                validation_losses=summary['val_loss'],
                title=f'{self.model_directory} - Learning Curve',
                path=settings.MODELS / self.model_directory / 'single_split' / f'learning_curve_seed{seed}.png'
            )

        visualization.visualize_predictions(
            y_true=df.loc[val_idx, self.target],
            y_pred=df.loc[val_idx, 'predictions'],
            title=f'{self.model_directory} - Single Split Predictions',
            path=settings.MODELS / self.model_directory / 'single_split' / 'predictions.png'
        )

    def train_no_split(self, df):

        """
        Train on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, n_columns)]: Dataframe of features and target
        """

        print(f'\n{"-" * 30}\nRunning Neural Network Model for Training\n{"-" * 30}\n')
        # Create directory to save models and training results
        Path(settings.MODELS / self.model_directory).mkdir(parents=True, exist_ok=True)

        train_dataset = TabularDataset(df.loc[:, self.features].values, df.loc[:, self.target].values)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_parameters['data_loader']['training_batch_size'],
            sampler=RandomSampler(train_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )

        for seed in self.seeds:

            print(f'\nSeed {seed}\n')

            # Set model, loss function, device and seed for reproducible results
            training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])
            criterion = getattr(torch_modules, self.training_parameters['loss_function'])(**self.training_parameters['loss_args'])
            model = getattr(torch_modules, self.model_parameters['model_class'])(**self.model_parameters['model_args'])
            if self.model_parameters['model_checkpoint_path'] is not None:
                model_checkpoint_path = self.model_parameters['model_checkpoint_path']
                model.load_state_dict(torch.load(model_checkpoint_path))
            model.to(device)

            # Set optimizer and learning rate scheduler
            optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_args']) if self.training_parameters['lr_scheduler'] is not None else None

            summary = {
                'train_loss': []
            }

            for epoch in range(1, self.training_parameters['epochs'] + 1):

                train_loss = self.train_fn(train_loader, model, criterion, optimizer, device, scheduler)

                print(f'Epoch {epoch} - Training Loss: {train_loss:.6f}')
                # Save model after every epoch
                model_path = settings.MODELS / self.model_directory / 'no_split' / f'model_seed{seed}.pt'
                torch.save(model.state_dict(), model_path)
                print(f'Saving model to {model_path}')

                summary['train_loss'].append(train_loss)

            # Visualize learning curve
            visualization.visualize_learning_curve(
                training_losses=summary['train_loss'],
                validation_losses=None,
                title=f'{self.model_directory} - Learning Curve',
                path=settings.MODELS / self.model_directory / 'no_split' / f'learning_curve_seed{seed}.png'
            )
