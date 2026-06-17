import os
import tempfile
import joblib
import hydra
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from common.data_manager import DataManager
from common.writer import MlflowWriter
from nam_family.classifier import grid_search, model_train


# python experiment_prefs.py --config-name nam dataset=breast experiment_name=test
@hydra.main(version_base=None, config_path='config', config_name='nam')
def main(cfg: DictConfig) -> None:

    print(f'Working directory : {os.getcwd()}')
    print(f'Dataset Name: {cfg.dataset}')
    dataset = DataManager(cfg.dataset, scalar='Standard', test_size=0.2, seed=0, downloads_path='./dataset/downloads/',
                          npz_path='./dataset/npz/')

    # Pre Feature Selection
    n_selected_feature = cfg.n_selected_feature
    print(f'Select {n_selected_feature} features out of {dataset.n_features}...')
    f_selector = SelectKBest(mutual_info_classif, k=n_selected_feature)
    f_selector.fit(dataset.x_train_scaled, dataset.y_train)
    fs_index = f_selector.get_support(indices=True)
    # Replace dataset
    dataset.x_train = f_selector.transform(dataset.x_train)
    dataset.x_train_scaled = f_selector.transform(dataset.x_train_scaled)
    dataset.x_test = f_selector.transform(dataset.x_test)
    dataset.x_test_scaled = f_selector.transform(dataset.x_test_scaled)
    dataset.n_features = n_selected_feature
    dataset.feature_names = dataset.feature_names[fs_index]

    writer = MlflowWriter(cfg.experiment_name)

    # Create mlflow run
    tags = {'mlflow.runName': 'experimental_setting'}
    writer.create_new_run(tags)
    # Save Config
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / 'config.yaml'
        OmegaConf.save(config=cfg, f=artifact_path)
        writer.log_artifact(artifact_path)

    # Feature Selector Saving
    with tempfile.TemporaryDirectory() as tmp_dir:
        fsel_save_path = Path(tmp_dir) / 'f_selector.joblib'
        joblib.dump(f_selector, fsel_save_path)
        writer.log_artifact(fsel_save_path)
    writer.set_terminated()

    # Training and Validation splitting
    x_train, x_valid, y_train, y_valid = train_test_split(dataset.x_train_scaled, dataset.y_train,
                                                          test_size=cfg.validation_rate, random_state=None,
                                                          stratify=dataset.y_train)
    # Grid search
    print(f'********** Grid Search **********')
    dict_model = OmegaConf.to_object(cfg.model)
    dict_grid = OmegaConf.to_object(cfg.grid_search_space)
    best_param = grid_search(dict_grid, dict_model, writer, dataset.n_class, x_train, y_train, x_valid, y_valid,
                             gpu_id=cfg.gpu_id, valid_metric=cfg.valid_metric)

    # Model training with best hyperparameters
    res_test = []
    for i in range(cfg.run_num):
        print(f'********** {i + 1}-th training run **********')

        # Create mlflow run
        tags = {'mlflow.runName': f'run-{i + 1:02d}'}
        writer.create_new_run(tags)

        # Training and Validation splitting
        x_train, x_valid, y_train, y_valid = train_test_split(dataset.x_train_scaled, dataset.y_train,
                                                              test_size=cfg.validation_rate, random_state=None,
                                                              stratify=dataset.y_train)
        # Model training
        val_score, nam_clf = model_train(OmegaConf.create(best_param), writer, dataset.n_class, x_train, y_train,
                                         x_valid, y_valid, gpu_id=cfg.gpu_id, valid_metric=cfg.valid_metric)

        # Testing
        test_acc, test_auroc = nam_clf.test(dataset.x_test_scaled, dataset.y_test, writer=writer, gpu_id=cfg.gpu_id)
        res_test += [[i + 1, test_acc, test_auroc]]
        writer.set_terminated()
        print('\nFinish...')

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
