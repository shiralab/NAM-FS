import sys
import os
import time
import tempfile
import numpy as np
from pathlib import Path
from urllib.parse import urlparse

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy, BinaryAUROC, MulticlassAUROC


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64, 32), activation='relu', dropout_rate=0., batch_norm=True):
        super(MLP, self).__init__()

        self.in_dim, self.out_dim = in_dim, out_dim

        if activation == 'relu':
            act_func = nn.ReLU
        elif activation == 'tanh':
            act_func = nn.Tanh
        else:
            print('Error: Invalid activation function...')
            sys.exit(1)

        # hidden layers
        in_features = in_dim
        layers = []
        for num_hidden in hidden_dims:
            layers += [nn.Linear(in_features=in_features, out_features=num_hidden)]
            if batch_norm:
                layers += [nn.BatchNorm1d(num_features=num_hidden)]
            if dropout_rate > 0:
                layers += [nn.Dropout(p=dropout_rate)]
            layers += [act_func()]

            in_features = num_hidden

        # Output linear layer
        layers += [nn.Linear(in_features=in_features, out_features=out_dim)]
        self.layers = nn.Sequential(*layers)

    # x: shape=(minibatch, in_dim), return: shape=(minibatch, out_dim)
    def forward(self, x):
        return self.layers(x)


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


class MLPClassifier(object):
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

        self.model = MLP(in_dim, out_dim,
                         hidden_dims=tuple(cfg_model.hidden_dims),
                         activation=cfg_model.activation,
                         dropout_rate=cfg_model.dropout_rate,
                         batch_norm=cfg_model.batch_norm)

        # Training Configuration
        self.validation_period = cfg_model.validation_period
        self.batch_size = cfg_model.batch_size
        self.init_lr = cfg_model.init_lr
        self.lr_decay = cfg_model.lr_decay
        self.lr_decay_patience = cfg_model.lr_decay_patience
        self.weight_decay = cfg_model.weight_decay
        self.max_iteration = cfg_model.max_iteration
        self.early_stopping_patience = cfg_model.early_stopping_patience

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
                outputs = self.model(x)

                # For binary classification
                if self.model.out_dim == 1:
                    outputs = outputs.squeeze(dim=1)

                loss = self.task_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_value = loss.detach().cpu().numpy()
                train_loss += loss_value * labels.size(0)
                total += labels.size(0)
                self.train_metric.update(outputs, labels)
                ite += 1
                counter += 1
                pbar.update(1)

                # Display and logging training and validation information
                if (ite % self.validation_period == 0) or (ite == self.max_iteration):
                    pbar.close()
                    train_loss /= total
                    metric_train = self.train_metric.compute()
                    self.train_logging(epoch, ite, train_loss, metric_train, lr_scheduler.get_last_lr()[0],
                                       print_func=print, writer=writer)
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

        # load best valid model
        if writer is not None:
            print('Loading best validation model...')
            arti_path = Path(urlparse(writer.run.info.artifact_uri).path).relative_to(os.getcwd())
            file_name = Path(arti_path / 'best_valid_model.pth')
            self.model.load_state_dict(torch.load(file_name))

        return float(early_stopping.best_score)

    def train_logging(self, epoch, ite, train_loss, metric_train, lr, print_func=print, writer=None):
        print_func(f'[Iteration {ite}] [Epoch {epoch}] '
                   f'[Training Loss {train_loss:.5f}] [Training Metric ({self.train_metric_name}) {metric_train:.5f}]')
        if writer is not None:
            writer.log_metric_step('epoch', epoch, step=ite)
            writer.log_metric_step('train_loss', train_loss, step=ite)
            writer.log_metric_step(f'train_{self.train_metric_name}', metric_train, step=ite)
            writer.log_metric_step('lr', lr, step=ite)

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

    def predict(self, x, batch_size=1024, gpu_id=0):
        # Running device
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print('Predicting using GPU device: ', device)
        else:
            device = torch.device('cpu')
            print('Predicting using CPU')
        self.model.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        test_start_t = time.time()

        x_tensor = torch.tensor(x, dtype=torch.float32)
        dataset = TensorDataset(x_tensor.to(device))
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predict_list = []
        output_list = []

        self.model.eval()
        pbar = tqdm(test_loader)
        pbar.set_description('Predicting')
        for x, in pbar:
            with torch.no_grad():
                outputs = self.model(x)

                # For binary classification
                if self.model.out_dim == 1:
                    outputs = outputs.squeeze(dim=1)
                    predicted = (outputs > 0).int()
                else:
                    _, predicted = torch.max(outputs, 1)

                predict_list += list(predicted.cpu().numpy().flatten())
                output_list += list(outputs.cpu().numpy().flatten())
        pbar.close()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        test_time = time.time() - test_start_t

        print('predict_time: ', test_time)
        predict_np = np.array(predict_list)
        output_np = np.array(output_list).reshape(-1, self.model.out_dim)

        if self.model.out_dim == 1:
            prob = torch.sigmoid(torch.tensor(output_np)).numpy()
            predict_prob_np = np.concatenate([1. - prob, prob], axis=1)
        else:
            predict_prob_np = torch.softmax(torch.tensor(output_np), dim=1).numpy()

        return predict_np, predict_prob_np
