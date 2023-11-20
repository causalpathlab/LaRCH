# LaRCH (LAtent Representaton of Cellular Hierarchies)

## Model development *modelhub.py*
- [X] dense model
- [X] spike slab
- [X] tree spike slab
  - [sparse models], pip < 1, e.g., 0.1, 0.01., 0.5
  - [dense model], pip = 1
- [X] tree susie spike slab

## Evaluation
- [X] Normalized Mutual Information (NMI)

- [X] Node/topic -specific genes

## Setup

```
python setup.py build
python setup.py install
```

## Run

### Spike and Slab model (BALSAM)

```
spike_slab [-h] [--nLV NLV] [--EPOCHS EPOCHS] [--lr LR] [--bs BS] [--train_size TRAIN_SIZE]
          [--seed SEED] [--use_gpu USE_GPU]
          [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--data_file DATA_FILE]
          [--data_id DATA_ID]
```

```
-h, --help            show this help message and exit
--nLV NLV             User specified nLV
--EPOCHS EPOCHS       EPOCHS
--lr LR               learning_rate
--bs BS               Batch size
--train_size TRAIN_SIZE
                      set to 1 to use full dataset for training; set to 0.9 for
                      train(0.9)/test(0.1) split
--seed SEED           seed
--use_gpu USE_GPU     which GPU to use
--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                      interval to perform evalutions
--data_file DATA_FILE
                      filepath to h5ad file
--data_id DATA_ID     data id
```

### Tree Spike and Slab model

```
usage: tree_spike_slab [-h] [--tree_depth TREE_DEPTH] [--EPOCHS EPOCHS] [--lr LR] [--bs BS]
                       [--pip0 PIP0] [--kl_weight KL_WEIGHT] [--kl_weight_beta KL_WEIGHT_BETA]
                       [--a0 A0] [--train_size TRAIN_SIZE] [--seed SEED] [--use_gpu USE_GPU]
                       [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--data_file DATA_FILE]
                       [--data_id DATA_ID] [--out_dir OUT_DIR]
```

```
-h, --help            show this help message and exit
--tree_depth TREE_DEPTH
                      tree depth
--EPOCHS EPOCHS       EPOCHS
--lr LR               learning_rate
--bs BS               Batch size
--pip0 PIP0           pip0
--kl_weight KL_WEIGHT
                      weight for kl local term
--kl_weight_beta KL_WEIGHT_BETA
                      weight for global parameter beta in the kl term
--a0 A0               hyperparameter for dirichlet likelihood
--train_size TRAIN_SIZE
                      set to 1 to use full dataset for training; set to 0.9 for
                      train(0.9)/test(0.1) split
--seed SEED           seed
--use_gpu USE_GPU     which GPU to use
--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                      interval to perform evalutions
--data_file DATA_FILE
                      filepath to h5ad file
--data_id DATA_ID     data id
--out_dir OUT_DIR     directory for output files
```

## Data Simulaation

```
usage: sc_data_sim_rho [-h] [--N N] [--noise NOISE] [--depth DEPTH] [--seed SEED] [--bulk_file BULK_FILE] [--out_dir OUT_DIR]
```

```
-h, --help            show this help message and exit
--N N                 Number of cells
--noise NOISE         Noise parameter
--depth DEPTH         Read Depth
--seed SEED           seed
--bulk_file BULK_FILE
                      filepath to bulk expression file
--out_dir OUT_DIR     output directory for simulated data
```

## Run Inference Only
```
usage: run_inference [-h] [--model_file MODEL_FILE] [--use_gpu USE_GPU] [--tree_depth TREE_DEPTH] [--test_data_file TEST_DATA_FILE] [--out_dir OUT_DIR]
```

```
-h, --help            show this help message and exit
--model_file MODEL_FILE
                      path to model file
--use_gpu USE_GPU     which GPU to use
--tree_depth TREE_DEPTH
                      Depth of tree model (optional)
--test_data_file TEST_DATA_FILE
                      path to data file to perform inference on
--out_dir OUT_DIR     path to save output file
  ```