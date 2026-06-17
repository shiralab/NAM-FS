import sys
import os
import numpy as np
import pandas as pd
import time
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import ParameterGrid
from omegaconf import OmegaConf

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy, BinaryAUROC, MulticlassAUROC

from .nam import NAM, NAMFS
from .nbm import NBM, NBMFS


class NAMFamilyClassifier(object):
    def __init__(self, in_dim, class_num, cfg_model, val_metric='ACC'):

        # Classification loss
        self.class_num = class_num
        if self.class_num == 2:
            out_dim = 1
            self.task_loss = nn.BCEWithLogitsLoss(reduction='mean')
            self.task_loss_name = 'bce'
            self.train_metric = BinaryAccuracy(threshold=0.)
            self.train_metric_name = 'acc'
            if val_metric == 'ACC':
                self.val_metric = BinaryAccuracy(threshold=0.)
                self.val_metric_name = 'acc'
            elif val_metric == 'AUROC':
                self.val_metric = BinaryAUROC()
                self.val_metric_name = 'auroc'
        else:
            out_dim = self.class_num
            self.task_loss = nn.CrossEntropyLoss(reduction='mean')
            self.task_loss_name = 'ce'
            self.train_metric = MulticlassAccuracy()
            self.train_metric_name = 'acc'
            if val_metric == 'ACC':
                self.val_metric = MulticlassAccuracy()
                self.val_metric_name = 'acc'
            elif val_metric == 'AUROC':
                self.val_metric = MulticlassAUROC(num_classes=self.class_num)
                self.val_metric_name = 'auroc'

        # Model
        self.model_name = cfg_model.name
        if self.model_name == 'NAM':
            self.model = NAM(in_dim, out_dim,
                             hidden_dims=tuple(cfg_model.hidden_dims),
                             use_exu=cfg_model.use_exu,
                             dropout_rate=cfg_model.dropout_rate,
                             feature_dropout=cfg_model.feature_dropout,
                             batch_norm=cfg_model.batch_norm,
                             pairwise=cfg_model.pairwise)
        elif self.model_name == 'NAMFS':
            self.model = NAMFS(in_dim, out_dim,
                               one_input_shape_num=cfg_model.one_input_shape_num,
                               two_input_shape_num=cfg_model.two_input_shape_num,
                               feature_sel=cfg_model.feature_sel,
                               hidden_dims=tuple(cfg_model.hidden_dims),
                               use_exu=cfg_model.use_exu,
                               dropout_rate=cfg_model.dropout_rate,
                               feature_dropout=cfg_model.feature_dropout,
                               batch_norm=cfg_model.batch_norm)
        elif self.model_name == 'NBM':
            self.model = NBM(in_dim, out_dim,
                             bases_num=cfg_model.bases_num,
                             hidden_dims=tuple(cfg_model.hidden_dims),
                             dropout_rate=cfg_model.dropout_rate,
                             bases_dropout=cfg_model.bases_dropout,
                             batch_norm=cfg_model.batch_norm,
                             pairwise=cfg_model.pairwise)
        elif self.model_name == 'NBMFS':
            self.model = NBMFS(in_dim, out_dim,
                               bases_num=cfg_model.bases_num,
                               one_input_shape_num=cfg_model.one_input_shape_num,
                               two_input_shape_num=cfg_model.two_input_shape_num,
                               feature_sel=cfg_model.feature_sel,
                               hidden_dims=tuple(cfg_model.hidden_dims),
                               dropout_rate=cfg_model.dropout_rate,
                               bases_dropout=cfg_model.bases_dropout,
                               batch_norm=cfg_model.batch_norm)
        else:
            print('Error: Undefined model name...')
            sys.exit(1)

        # Training Configuration
        self.validation_period = cfg_model.validation_period
        self.batch_size = cfg_model.batch_size
        self.init_lr = cfg_model.init_lr
        self.lr_decay = cfg_model.lr_decay
        self.lr_decay_patience = cfg_model.lr_decay_patience
        self.weight_decay = cfg_model.weight_decay
        self.output_penalty = cfg_model.output_penalty
        self.max_iteration = cfg_model.max_iteration
        self.early_stopping_patience = cfg_model.early_stopping_patience

        if self.model_name == 'NAMFS' or self.model_name == 'NBMFS':
            self.annealing_iteration = cfg_model.annealing_iteration
            self.max_temperature = cfg_model.max_temperature
            self.min_temperature = cfg_model.min_temperature
            self.annealing_schedule = cfg_model.annealing_schedule
        else:
            self.annealing_iteration = 0
            self.max_temperature = 0
            self.min_temperature = 0
            self.annealing_schedule = 'linear'

    def train(self, x_train, y_train, x_valid, y_valid, writer=None, gpu_id=0, num_workers=0):
        # Running device
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print('Training using GPU device: ', device)
            self.train_metric = self.train_metric.to(device)
            self.val_metric = self.val_metric.to(device)
        else:
            device = torch.device('cpu')
            print('Training using CPU')
        self.model.to(device)

        # Time
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        test_start_t = time.time()

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        # Dataset
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
        if self.model.out_dim == 1:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_valid, dtype=torch.float32)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_val_tensor = torch.tensor(y_valid, dtype=torch.long)
        train_dataset = TensorDataset(x_train_tensor.to(device), y_train_tensor.to(device))
        val_dataset = TensorDataset(x_valid_tensor.to(device), y_val_tensor.to(device))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=num_workers)

        # Early stopping, LR decay, and temperature annealing
        early_stopping = ValidationMonitor(patience=self.early_stopping_patience, description='Early Stopping')
        lr_decay = ValidationMonitor(patience=self.lr_decay_patience, description='LR Decay')
        annealing_scheduler = TemperatureAnnealing(self.annealing_iteration, max_t=self.max_temperature,
                                                   min_t=self.min_temperature, schedule=self.annealing_schedule)

        # Training loop
        ite, epoch = 0, 0
        train_loss = 0.
        total = 0
        counter = 0
        best_valid_iteration = 1
        pbar = tqdm(total=self.validation_period)
        while ite < self.max_iteration:
            epoch += 1

            # Training epoch
            self.train_metric.reset()
            for x, labels in train_loader:
                self.model.train()
                optimizer.zero_grad()
                if ite < self.annealing_iteration:
                    outputs = self.model(x, temperature=annealing_scheduler.temperature)
                else:
                    outputs = self.model(x)

                # For binary classification
                if self.model.out_dim == 1:
                    outputs = outputs.squeeze(dim=1)

                loss = self.task_loss(outputs, labels)
                if self.output_penalty > 0.:
                    loss += self.output_penalty * torch.mean(self.model.feature_outputs ** 2, dim=-1).mean()
                loss.backward()
                optimizer.step()

                loss_value = loss.detach().cpu().numpy()
                train_loss += loss_value * labels.size(0)
                total += labels.size(0)
                self.train_metric.update(outputs, labels)
                ite += 1
                counter += 1
                pbar.update(1)

                # Temperature annealing step
                annealing_scheduler(1)

                # Display and logging training and validation information
                if (ite % self.validation_period == 0) or (ite == self.max_iteration):
                    pbar.close()
                    train_loss /= total
                    metric_train = self.train_metric.compute()
                    self.train_logging(epoch, ite, train_loss, metric_train, lr_scheduler.get_last_lr()[0],
                                       annealing_scheduler.temperature, print_func=print, writer=writer)
                    train_loss = 0.
                    total = 0

                    # Run validation
                    valid_metric = self.validation(val_loader, ite, print_func=print, writer=writer)

                    # Early stopping step
                    early_stopping(valid_metric, counter)
                    # Early stopping
                    if early_stopping.over:
                        print('[Finished] early stopping')
                        break

                    # LR decay step
                    lr_decay(valid_metric, counter)
                    if lr_decay.over:
                        print('Decay learning rate')
                        lr_scheduler.step()
                        lr_decay.reset_counter()

                    # Save best validation model
                    if early_stopping.counter == 0:
                        best_valid_iteration = ite
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            artifact_path = Path(tmp_dir) / 'best_valid_model.pth'
                            torch.save(self.model.state_dict(), artifact_path)
                            if writer is not None:
                                writer.log_artifact(artifact_path)

                    counter = 0
                    pbar = tqdm(total=self.validation_period)

                if ite >= self.max_iteration:
                    pbar.close()
                    print('[Finished] reach maximum number of iterations')
                    break

            if early_stopping.over:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        train_time = time.time() - test_start_t
        print(f'\nterminate_iteration: {ite}, train_time: {train_time}, '
              f'best_valid_metric: {early_stopping.best_score}, '
              f'best_valid_iteration: {best_valid_iteration}')
        if writer is not None:
            writer.log_metric('terminate_iteration', ite)
            writer.log_metric('train_time', train_time)
            writer.log_metric('best_valid_metric', early_stopping.best_score)
            writer.log_metric('best_valid_iteration', best_valid_iteration)

        # Load best valid model
        if writer is not None:
            print('Loading best validation model...')
            arti_path = Path(urlparse(writer.run.info.artifact_uri).path).relative_to(os.getcwd())
            file_name = Path(arti_path / 'best_valid_model.pth')
            self.model.load_state_dict(torch.load(file_name))

        return float(early_stopping.best_score)

    def train_logging(self, epoch, ite, train_loss, metric_train, lr, temperature, print_func=print, writer=None):
        print_func(f'[Iteration {ite}] [Epoch {epoch}] '
                   f'[Training Loss {train_loss:.5f}] [Training Metric ({self.train_metric_name}) {metric_train:.5f}] '
                   f'[LR {lr:.5f}] [Temperature {temperature:.5f}]')
        if writer is not None:
            writer.log_metric_step('epoch', epoch, step=ite)
            writer.log_metric_step('train_loss', train_loss, step=ite)
            writer.log_metric_step(f'train_{self.train_metric_name}', metric_train, step=ite)
            writer.log_metric_step('lr', lr, step=ite)
            writer.log_metric_step('temperature', temperature, step=ite)

    def validation(self, val_loader, ite, print_func=print, writer=None):
        self.model.eval()
        self.val_metric.reset()
        val_task_loss = 0.
        total = 0
        for x, labels in val_loader:
            with torch.no_grad():
                outputs = self.model(x)

                # For binary classification
                if self.model.out_dim == 1:
                    outputs = outputs.squeeze(dim=1)

                val_task_loss += self.task_loss(outputs, labels).detach().cpu().numpy() * labels.size(0)
                total += labels.size(0)
                self.val_metric.update(outputs, labels)
        val_task_loss /= total
        valid_metric = self.val_metric.compute()

        print_func(f'[Iteration {ite}] [Validation Task Loss ({self.task_loss_name}) {val_task_loss:.5f}] '
                   f'[Validation Metric ({self.val_metric_name}) {valid_metric:.5f}]')
        if writer is not None:
            writer.log_metric_step(f'val_{self.task_loss_name}', val_task_loss, step=ite)
            writer.log_metric_step(f'val_{self.val_metric_name}', valid_metric, step=ite)

        return valid_metric

    def test(self, x_test, y_test, batch_size=1024, writer=None, gpu_id=0):
        # Running device
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print('Testing using GPU device: ', device)
        else:
            device = torch.device('cpu')
            print('Testing using CPU')
        self.model.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        test_start_t = time.time()

        x_tensor = torch.tensor(x_test, dtype=torch.float32)
        if self.model.out_dim == 1:
            y_tensor = torch.tensor(y_test, dtype=torch.float32)
        else:
            y_tensor = torch.tensor(y_test, dtype=torch.long)
        dataset = TensorDataset(x_tensor.to(device), y_tensor.to(device))
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Performance metric
        if self.model.out_dim == 1:
            acc = BinaryAccuracy(threshold=0., device=device).reset()
            au_roc = BinaryAUROC(device=device).reset()
        else:
            acc = MulticlassAccuracy(device=device).reset()
            au_roc = MulticlassAUROC(num_classes=self.class_num, device=device).reset()

        target_list = []
        predict_list = []
        output_list = []

        self.model.eval()
        pbar = tqdm(test_loader)
        pbar.set_description('Testing')
        for x, labels in pbar:
            with torch.no_grad():
                outputs = self.model(x)

                # For binary classification
                if self.model.out_dim == 1:
                    outputs = outputs.squeeze(dim=1)
                    predicted = (outputs > 0).int()
                else:
                    _, predicted = torch.max(outputs, 1)

                acc.update(outputs, labels)
                au_roc.update(outputs, labels)

                target_list += list(labels.cpu().numpy().flatten())
                predict_list += list(predicted.cpu().numpy().flatten())
                output_list += list(outputs.cpu().numpy().flatten())
        pbar.close()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        test_time = time.time() - test_start_t

        acc_val = acc.compute()
        au_roc_val = au_roc.compute()

        print('test_time: ', test_time)
        print('test_acc: ', float(acc_val))
        print('test_au_roc: ', float(au_roc_val))
        if writer is not None:
            writer.log_metric('test_time', test_time)
            writer.log_metric('test_acc', acc_val)
            writer.log_metric('test_au_roc', au_roc_val)

        dic = {'Target': target_list, 'Predict': predict_list}
        output_np = np.array(output_list).reshape(-1, self.model.out_dim)
        for i in range(self.model.out_dim):
            dic[f'Output{i}'] = list(output_np[:, i])
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / 'test_inference.csv'
            pd.DataFrame(dic).to_csv(artifact_path, index=False)
            if writer is not None:
                writer.log_artifact(artifact_path)

        return float(acc_val), float(au_roc_val)


class ValidationMonitor:
    """ Monitoring validation score for early stopping, etc. """
    def __init__(self, patience=100, delta=1e-10, description='Early Stopping'):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            description (str): Description of Monitoring
        """
        self.patience = patience
        self.delta = delta
        self.description = description

        self.counter = 0
        self.best_score = None
        self.over = False

    def reset_counter(self):
        self.counter = 0
        self.over = False

    def __call__(self, val_score, increment_count, trace_func=print):
        """
            trace_func (function): trace print function.  Default: print
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:  # if validation score is not improved
            self.counter += increment_count
            trace_func(f'{self.description} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.over = True
        else:  # if validation score is improved
            trace_func(f'{self.description} counter: validation score is improved'
                       f' ({self.best_score:.5f} --> {val_score:.5f})')
            self.best_score = val_score
            self.counter = 0
        return self.over


class TemperatureAnnealing:
    def __init__(self, annealing_iteration=100, max_t=1., min_t=0., schedule='linear'):
        self.annealing_iteration = annealing_iteration
        self.max_t = max_t
        self.min_t = min_t
        self.schedule = schedule

        self.counter = 0
        self.temperature = max_t

    def __call__(self, increment_count):
        self.counter += increment_count
        if self.counter > self.annealing_iteration:
            self.temperature = 0
        else:
            # Linear
            if self.schedule == 'linear':
                self.temperature = self.max_t - (self.counter / self.annealing_iteration) * (self.max_t - self.min_t)
            # Exponential
            elif self.schedule == 'exponential':
                self.temperature = self.max_t * (self.min_t / self.max_t) ** (self.counter / self.annealing_iteration)
            else:
                print('Error: invalid temperature annealing schedule...')
                sys.exit(1)

        return self.temperature


def model_train(cfg_model, writer, class_num, x_train, y_train, x_valid, y_valid, gpu_id=0, valid_metric='ACC'):
    # Save configuration
    if writer is not None:
        writer.log_params_from_omegaconf_dict(cfg_model)
        writer.log_param('valid_metric', valid_metric)
        # Save config yaml
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / 'config.yaml'
            OmegaConf.save(config=cfg_model, f=artifact_path)
            writer.log_artifact(artifact_path)

    # Train
    print(f'Num. of input dimension: {x_train.shape[1]}')
    print(f'Num. of classes: {class_num}\n')
    nam_clf = NAMFamilyClassifier(x_train.shape[1], class_num, cfg_model, val_metric=valid_metric)
    val_score = nam_clf.train(x_train, y_train, x_valid, y_valid, gpu_id=gpu_id, writer=writer)
    return val_score, nam_clf


def grid_search(dict_grid, dict_model, writer, class_num, x_train, y_train, x_valid, y_valid, gpu_id=0,
                valid_metric='ACC'):

    best_val_score = None
    best_grid_param = None
    best_param = None
    res_dict = {}
    dict_m = dict_model.copy()

    with tempfile.TemporaryDirectory() as tmp_dir:

        gs_start_t = time.time()

        for i, param in enumerate(ParameterGrid(dict_grid)):
            # Set configuration
            print(f'\n{i}-th hyper-parameters:')
            dict_m.update(param)
            print(dict_m)

            # Create mlflow run
            tags = {'mlflow.runName': f'grid_search:{i:03d}'}
            writer.create_new_run(tags)

            # Model training
            val_score, nam_clf = model_train(OmegaConf.create(dict_m), writer, class_num, x_train, y_train, x_valid,
                                             y_valid, gpu_id=gpu_id, valid_metric=valid_metric)

            # Add result to dictionary
            if len(res_dict) == 0:
                for k in dict_m:
                    res_dict[k] = [dict_m[k]]
                res_dict['valid_score'] = [val_score]
            else:
                for k in dict_m:
                    res_dict[k] += [dict_m[k]]
                res_dict['valid_score'] += [val_score]

            # Save best validation information
            if (best_val_score is None) or (best_val_score < val_score):
                best_id = i
                best_val_score = val_score
                best_grid_param = param.copy()
                best_param = dict_m.copy()

                # Update best model file
                model_save_path = Path(tmp_dir) / 'best_gs_model.pth'
                torch.save(nam_clf.model.state_dict(), model_save_path)

            # Terminate mlflow run
            writer.set_terminated()

        gs_time = time.time() - gs_start_t

        # Create mlflow run
        tags = {'mlflow.runName': 'grid_search_result'}
        writer.create_new_run(tags)

        # Save grid search result
        print('grid_search_time: ', gs_time)
        writer.log_metric('grid_search_time', gs_time)
        writer.log_metric('best_validation_score', best_val_score)
        writer.log_metric('best_grid_id', best_id)
        writer.log_artifact(model_save_path)

        artifact_path = Path(tmp_dir) / 'best_config.yaml'
        OmegaConf.save(config=best_param, f=artifact_path)
        writer.log_artifact(artifact_path)

        artifact_path = Path(tmp_dir) / 'best_grid.yaml'
        OmegaConf.save(config=best_grid_param, f=artifact_path)
        writer.log_artifact(artifact_path)

        artifact_path = Path(tmp_dir) / 'grid_search_result.csv'
        pd.DataFrame(res_dict).to_csv(artifact_path)
        writer.log_artifact(artifact_path)

        # Terminate mlflow run
        writer.set_terminated()

    print('\nBest configuration: ', best_param)
    print('\nBest grid parameter: ', best_grid_param)
    print('\nBest validation score: ', best_val_score)

    return best_param
