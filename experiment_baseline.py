import os
import sys
import time
import tempfile
import joblib
import hydra
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, ParameterGrid
from torcheval.metrics import BinaryAUROC, MulticlassAUROC

from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from nodegam.sklearn import NodeGAMClassifier
from mlp.model import MLPClassifier

from common.data_manager import DataManager
from common.writer import MlflowWriter


# python experiment_baseline.py --config-name linear dataset=breast experiment_name=test
@hydra.main(version_base=None, config_path='config', config_name='nam')
def main(cfg: DictConfig) -> None:

    print(f'Working directory : {os.getcwd()}')
    print(f'Dataset Name: {cfg.dataset}')
    dataset = DataManager(cfg.dataset, scalar='Standard', test_size=0.2, seed=0, downloads_path='./dataset/downloads/',
                          npz_path='./dataset/npz/')

    writer = MlflowWriter(cfg.experiment_name)

    # Create mlflow run
    tags = {'mlflow.runName': 'experimental_setting'}
    writer.create_new_run(tags)
    # Save Config
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / 'config.yaml'
        OmegaConf.save(config=cfg, f=artifact_path)
        writer.log_artifact(artifact_path)
    writer.set_terminated()

    # Training and Validation splitting
    x_train, x_valid, y_train, y_valid = train_test_split(dataset.x_train_scaled, dataset.y_train,
                                                          test_size=cfg.validation_rate, random_state=None,
                                                          stratify=dataset.y_train)

    # Grid search
    print(f'\n********** Grid Search **********')
    model_params_dict = OmegaConf.to_object(cfg.model)
    best_val_score = None
    best_param = None
    res_dict = {}

    if len(ParameterGrid(model_params_dict)) > 1:
        gs_start_t = time.time()
        for i, param in enumerate(ParameterGrid(model_params_dict)):
            # Set configuration
            print(f'{i+1}-th hyper-parameters ({i+1}/{len(ParameterGrid(model_params_dict))}):')
            print(param)
            start_t = time.time()

            # ********** HERE **********
            if cfg.name == 'LR':
                if param['penalty'] == 'l1':
                    solver = 'saga'
                else:
                    solver = 'lbfgs'
                clf = LogisticRegression(**param, solver=solver)
                clf.fit(x_train, y_train)
                predict_valid = clf.predict(x_valid)

            elif cfg.name == 'DT':
                clf = DecisionTreeClassifier(**param)
                clf.fit(x_train, y_train)
                predict_valid = clf.predict(x_valid)

            elif cfg.name == 'XGB':
                seed = np.random.randint(3000)
                clf = XGBClassifier(**param, random_state=seed)
                clf.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
                        verbose=False)
                predict_valid = clf.predict(x_valid)

            elif cfg.name == 'EBM':
                seed = np.random.randint(3000)
                clf = ExplainableBoostingClassifier(**param, random_state=seed)
                clf.fit(x_train, y_train)
                predict_valid = clf.predict(x_valid)

            elif cfg.name == 'NODE-GAM':
                if dataset.n_class == 2:
                    n_class_node = 1
                else:
                    n_class_node = dataset.n_class
                seed = np.random.randint(3000)
                clf = NodeGAMClassifier(**param, in_features=x_train.shape[1], num_classes=n_class_node, seed=seed,
                                        verbose=True)
                train_record = clf.fit(pd.DataFrame(x_train), y_train)
                predict_proba_valid = clf.predict_proba(pd.DataFrame(x_valid))
                predict_valid = np.argmax(predict_proba_valid, axis=1)

            elif cfg.name == 'MLP':
                # Create mlflow run
                tags = {'mlflow.runName': f'grid_search:{i:03d}'}
                writer.create_new_run(tags)

                clf = MLPClassifier(dataset.n_features, dataset.n_class, OmegaConf.create(param), val_metric='ACC')
                val_score = clf.train(x_train, y_train, x_valid, y_valid, writer=writer, gpu_id=cfg.gpu_id)
                predict_valid, _ = clf.predict(x_valid, batch_size=1024, gpu_id=cfg.gpu_id)
                writer.set_terminated()

            else:
                print('Error: Invalid model name!')
                sys.exit(1)

            # Valid metric (ACC only: TODO)
            val_score = accuracy_score(y_valid, predict_valid)
            print(f'Validation ACC: {val_score}')

            # Add result to dictionary
            if len(res_dict) == 0:
                for k in param:
                    res_dict[k] = [param[k]]
                res_dict['valid_score'] = [val_score]
            else:
                for k in param:
                    res_dict[k] += [param[k]]
                res_dict['valid_score'] += [val_score]

            # Save best validation information
            if (best_val_score is None) or (best_val_score < val_score):
                best_val_score = val_score
                best_param = param.copy()

            print(f'Time: {(time.time() - start_t) / 60.:.3f} (min.)\n')

        gs_time = time.time() - gs_start_t
        # Create mlflow run
        tags = {'mlflow.runName': 'grid_search_result'}
        writer.create_new_run(tags)
        # Save grid search result
        print('\ngrid_search_time: ', gs_time)
        print('Best grid parameter: ', best_param)
        print('Best score: ', best_val_score, '\n')
        writer.log_metric('grid_search_time', gs_time)
        writer.log_metric('best_score', best_val_score)
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / 'best_config.yaml'
            OmegaConf.save(config=best_param, f=artifact_path)
            writer.log_artifact(artifact_path)

            artifact_path = Path(tmp_dir) / 'grid_search_result.csv'
            pd.DataFrame(res_dict).to_csv(artifact_path)
            writer.log_artifact(artifact_path)
        # Terminate mlflow run
        writer.set_terminated()

    else:
        print('No need to perform grid search...\n')
        best_param = ParameterGrid(model_params_dict)[0]

    # Model training with best hyperparameters
    res_test = []
    for i in range(cfg.run_num):
        # Create mlflow run
        tags = {'mlflow.runName': f'run-{i + 1:02d}'}
        writer.create_new_run(tags)
        writer.log_params_from_omegaconf_dict(best_param)

        # Save config yaml
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / 'config.yaml'
            OmegaConf.save(config=best_param, f=artifact_path)
            writer.log_artifact(artifact_path)

        print(f'********** {i + 1}-th training run **********')
        start_t = time.time()
        print(best_param)

        # Training and Validation splitting
        x_train, x_valid, y_train, y_valid = train_test_split(dataset.x_train_scaled, dataset.y_train,
                                                              test_size=cfg.validation_rate, random_state=None,
                                                              stratify=dataset.y_train)

        # ********** HERE **********
        if cfg.name == 'LR':
            if best_param['penalty'] == 'l1':
                solver = 'saga'
            else:
                solver = 'lbfgs'
            clf = LogisticRegression(**best_param, solver=solver)
            clf.fit(dataset.x_train_scaled, dataset.y_train)
            predict_test = clf.predict(dataset.x_test_scaled)
            predict_proba_test = clf.predict_proba(dataset.x_test_scaled)

        elif cfg.name == 'DT':
            clf = DecisionTreeClassifier(**best_param)
            clf.fit(dataset.x_train_scaled, dataset.y_train)
            predict_test = clf.predict(dataset.x_test_scaled)
            predict_proba_test = clf.predict_proba(dataset.x_test_scaled)

        elif cfg.name == 'XGB':
            seed = np.random.randint(3000)
            writer.log_param('seed', seed)
            clf = XGBClassifier(**best_param, random_state=seed)
            clf.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
            predict_test = clf.predict(dataset.x_test_scaled)
            predict_proba_test = clf.predict_proba(dataset.x_test_scaled)

        elif cfg.name == 'EBM':
            seed = np.random.randint(3000)
            writer.log_param('seed', seed)
            clf = ExplainableBoostingClassifier(**best_param, random_state=seed)
            clf.fit(dataset.x_train_scaled, dataset.y_train)
            predict_test = clf.predict(dataset.x_test_scaled)
            predict_proba_test = clf.predict_proba(dataset.x_test_scaled)

        elif cfg.name == 'NODE-GAM':
            if dataset.n_class == 2:
                n_class_node = 1
            else:
                n_class_node = dataset.n_class
            seed = np.random.randint(3000)
            writer.log_param('seed', seed)
            clf = NodeGAMClassifier(**best_param, in_features=dataset.x_train.shape[1], num_classes=n_class_node,
                                    seed=seed, verbose=True)
            train_record = clf.fit(pd.DataFrame(dataset.x_train_scaled), dataset.y_train)
            predict_proba_test = clf.predict_proba(pd.DataFrame(dataset.x_test_scaled))
            predict_test = np.argmax(predict_proba_test, axis=1)

        elif cfg.name == 'MLP':
            clf = MLPClassifier(dataset.n_features, dataset.n_class, OmegaConf.create(best_param), val_metric='ACC')
            val_score = clf.train(x_train, y_train, x_valid, y_valid, writer=writer, gpu_id=cfg.gpu_id)
            predict_test, predict_proba_test = clf.predict(dataset.x_test_scaled, batch_size=1024, gpu_id=cfg.gpu_id)

        else:
            print('Error: Invalid model name!')
            sys.exit(1)

        train_time = time.time() - start_t
        writer.log_metric('train_time', train_time)

        # Model Saving
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_save_path = Path(tmp_dir) / 'trained_model.joblib'
            joblib.dump(clf, model_save_path)
            writer.log_artifact(model_save_path)

        # Test metric
        test_acc = accuracy_score(dataset.y_test, predict_test)
        if dataset.n_class == 2:
            test_auroc = BinaryAUROC().update(torch.tensor(predict_proba_test[:, 1]),
                                              torch.tensor(dataset.y_test)).compute()
        else:
            test_auroc = MulticlassAUROC(num_classes=dataset.n_class).update(torch.tensor(predict_proba_test),
                                                                             torch.tensor(dataset.y_test)).compute()
        res_test += [[i + 1, float(test_acc), float(test_auroc)]]

        print('test_acc: ', float(test_acc))
        print('test_au_roc: ', float(test_auroc))
        writer.log_metric('test_acc', float(test_acc))
        writer.log_metric('test_au_roc', float(test_auroc))

        dic = {'Target': list(dataset.y_test.flatten()), 'Predict_Proba': list(predict_test.flatten())}
        for j in range(predict_proba_test.shape[1]):
            dic[f'Predict_Proba{j}'] = list(predict_proba_test[:, j])
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / 'test_inference.csv'
            pd.DataFrame(dic).to_csv(artifact_path, index=False)
            writer.log_artifact(artifact_path)

        writer.set_terminated()
        print('Finish...\n')

    df_test = pd.DataFrame(res_test, columns=['training_run_id', 'test_acc', 'test_auroc'])
    print(f'[Test ACC] mean: {df_test["test_acc"].mean()}, std: {df_test["test_acc"].std()}, '
          f'max: {df_test["test_acc"].max()}, min: {df_test["test_acc"].min()}')
    print(f'[Test AUROC] mean: {df_test["test_auroc"].mean()}, std: {df_test["test_acc"].std()}, '
          f'max: {df_test["test_auroc"].max()}, min: {df_test["test_acc"].min()}')

    # Create mlflow run for summary
    tags = {'mlflow.runName': 'experimental_result'}
    writer.create_new_run(tags)

    writer.log_metric('test_acc_ave', df_test['test_acc'].mean())
    writer.log_metric('test_acc_std', df_test['test_acc'].std())
    writer.log_metric('test_acc_max', df_test['test_acc'].max())
    writer.log_metric('test_acc_min', df_test['test_acc'].min())
    writer.log_metric('test_auroc_ave', df_test['test_auroc'].mean())
    writer.log_metric('test_auroc_std', df_test['test_auroc'].std())
    writer.log_metric('test_auroc_max', df_test['test_auroc'].max())
    writer.log_metric('test_auroc_min', df_test['test_auroc'].min())

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / 'test_result.csv'
        df_test.to_csv(artifact_path, index=False)
        writer.log_artifact(artifact_path)
    writer.set_terminated()


if __name__ == "__main__":
    main()
