# Neural Additive and Basis Models with Feature Selection and Interactions

This repository provides the code for the following paper.

- Yasutoshi Kishimoto, Kota Yamanishi, Takuya Matsuda, and Shinichi Shirakawa, "Neural Additive and Basis Models with Feature Selection and Interactions," *PAKDD 2024*.

## Directory
```
.
├── common
├── config
├── mlp
├── experiment_baseline.py
├── experiment_nam.py
├── experiment_prefs.py
└── nam_family

```

- __common__
    - codes for dataset preparation and logging

- __config__
    - config files for experiments

- __mlp__
    - code for MLP

- __experiment_baseline.py__
  - main script for the experiments of baseline models

- __experiment_nam.py__
  - main script for the experiments of NAM and NBM families

- __experiment_prefs.py__
  - main script for the experiments of NAM and NBM with pre-selected features

- __nam_family__
    - code for NAM and NBM families, including our proposed models (NAM-FS and NBM-FS)


## Requirement
Please create an conda environment using environment.yml. 
```
conda env create --file environment.yml
conda activate nam_proj
```

## Usage

### Preparation
When using the Epsilon and Fashion MNIST datasets, please locate the downloaded files at `./dataset/downloads`.

### Run
- Examples for running experiments for NAM and NBM families
    - DATASET_NAME: {har, isolet, fashion-mnist, epsilon, guillermo, gisette}
    ```shell
    # NAM
    python experiment_nam.py --config-name nam dataset=DATASET_NAME experiment_name=NAM gpu_id=0

    # NAM-FS (K_1=500)
    python experiment_nam.py --config-name namfs dataset=DATASET_NAME experiment_name=NAMFS-500 model.one_input_shape_num=500 model.two_input_shape_num=0 gpu_id=0

    # NA^2M-FS (K_1=K_2=500)
    python experiment_nam.py --config-name namfs dataset=DATASET_NAME experiment_name=NA2MFS-500 model.one_input_shape_num=500 model.two_input_shape_num=500 gpu_id=0

    # NBM
    python experiment_nam.py --config-name nbm dataset=DATASET_NAME experiment_name=NBM gpu_id=0

    # NBM-FS (K_1=500)
    python experiment_nam.py --config-name nbmfs dataset=DATASET_NAME experiment_name=NBMFS-500 model.one_input_shape_num=500 model.two_input_shape_num=0 gpu_id=0

    # NB^2M-FS (K_1=K_2=500)
    python experiment_nam.py --config-name nbmfs dataset=DATASET_NAME experiment_name=NB2MFS-500 model.one_input_shape_num=500 model.two_input_shape_num=500 model.bases_num=200 gpu_id=0
    ```

- Examples of running experiments for baseline models
    ```shell
    # EBM
    python experiment_baseline.py --config-name ebm dataset=DATASET_NAME experiment_name=EBM

    # EB^2M (only for binary classification)
    python experiment_baseline.py --config-name ebm dataset=DATASET_NAME experiment_name=EB2M "model.interactions=[16,64,128,512]"

    # NODE-GAM
    python experiment_baseline.py --config-name nodegam dataset=DATASET_NAME experiment_name=NODE-GAM "model.device=[cuda:0]" "+model.name=[temperory_name]"

    # NODE-GA^2M
    python experiment_baseline.py --config-name nodegam dataset=DATASET_NAME experiment_name=NODE-GA2M "model.ga2m=[1]" "model.device=[cuda:0]" "+model.name=[temperory_name]"

    # Logistic Regression (LR)
    python experiment_baseline.py --config-name linear dataset=DATASET_NAME experiment_name=LR

    # Decision Tree (DT)
    python experiment_baseline.py --config-name dt dataset=DATASET_NAME experiment_name=DT

    # MLP
    python experiment_baseline.py --config-name mlp dataset=DATASET_NAME experiment_name=MLP gpu_id=0

    # XGBoost
    python experiment_baseline.py --config-name xgb dataset=DATASET_NAME experiment_name=XGB
    ```

- Examples of running experiments for NAM and NBM with pre-selected features
    ```shell
    # NAM with pre-selected features
    python experiment_prefs.py --config-name nam dataset=DATASET_NAME experiment_name=NAM-PreFS-50n gpu_id=0 +n_selected_feature=50

    # NBM with pre-selected features
    python experiment_prefs.py --config-name nbm dataset=DATASET_NAME experiment_name=NBM-PreFS-50n gpu_id=0 +n_selected_feature=50
    ```

## Reference
Yasutoshi Kishimoto, Kota Yamanishi, Takuya Matsuda, and Shinichi Shirakawa, "Neural Additive and Basis Models with Feature Selection and Interactions," *Proceedings of the 28th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2024), Part III*, Vol. 14647 of LNAI, pp. 3-16, 2024. [[DOI](https://doi.org/10.1007/978-981-97-2259-4_1)]
