# LaRCH (LAtent Representaton of Cellular Hierarchies)

## Data
- [X] Simulation
  - High-dimensional case: N = 5000, G = 10000, S = 50, D_tree = 5
  - Low-dimensional case: N = 5000, G = 1000, S = 20, D_tree = 5

## Model development *modelhub.py*
- [ ] dense model
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

### Spike and Slab model

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

### SuSiE tree model

```
tree_susie [-h] [--tree_depth TREE_DEPTH] [--pip0 PIP0] [--EPOCHS EPOCHS] [--lr LR]
          [--bs BS] [--kl_weight KL_WEIGHT] [--kl_weight_beta KL_WEIGHT_BETA]
          [--train_size TRAIN_SIZE] [--seed SEED] [--use_gpu USE_GPU]
          [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--data_file DATA_FILE]
          [--data_id DATA_ID]
```

```
-h, --help            show this help message and exit
--tree_depth TREE_DEPTH
                      tree depth
--pip0 PIP0           pip0
--EPOCHS EPOCHS       EPOCHS
--lr LR               learning_rate
--bs BS               Batch size
--kl_weight KL_WEIGHT
                      weight for kl local term
--kl_weight_beta KL_WEIGHT_BETA
                      weight for global parameter beta in the kl term
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
