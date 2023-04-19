import os
import scanpy as sc
import pandas as pd
import argparse
import torch
import pickle
from larch.util.modelhub import tree_spike_slab

def main():
    parser = argparse.ArgumentParser(description='Inference parameters')
    parser.add_argument('--model_file', help='path to model file', default='model_params.pt')
    parser.add_argument('--test_data_file', help='path to data file to perform inference on', default='data/sim_data.h5ad')
    parser.add_argument('--out_dir', help='path to save output file', default='data')

    args = parser.parse_args()
    print(args)

    test_data = sc.read(args.test_data_file)
    model = tree_spike_slab()
    with open(args.model_file, "rb") as fp:
        model.load_state_dict(pickle.load(fp))

    test_theta = model.get_latent_representation(adata, deterministic=True, output_softmax_z=True)

    print("---Saving topic proportions (after softmax)---\n")
    topics_df = pd.DataFrame(test_theta, index= test_data.obs.index, columns = ['topic_' + str(j) for j in range(topics_np.shape[1])])
    topics_df.to_csv(os.path.join(args.out_dir, model_id,"topics.csv"))

if __name__ == "__main__":
    main()
