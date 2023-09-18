import os
import scanpy as sc
import pandas as pd
import argparse
import torch
import pickle
from scipy.sparse import csr_matrix
from larch.util.modelhub import TreeSpikeSlab
from larch.util.util import setup_anndata

def main():
    parser = argparse.ArgumentParser(description='Inference parameters')
    parser.add_argument('--model_file', help='path to model file', default='model_params.pt')
    parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--tree_depth', help='Depth of tree model (optional)', type = int, required=False)
    parser.add_argument('--test_data_file', help='path to data file to perform inference on', default='data/sim_data.h5ad')
    parser.add_argument('--out_dir', help='path to save output file', default='data')

    args = parser.parse_args()
    print(args)

    if not args.tree_depth:
        if not os.path.dirname(args.model_file):
            model_dir = os.getcwd()
        else:
            model_dir = os.path.dirname(args.model_file)
        tree_depth = int(model_dir.split("treeD")[1].split("_")[0])
    else: tree_depth = args.tree_depth

    test_data = sc.read(args.test_data_file)
    test_data.layers["counts"] = csr_matrix(test_data.X).copy()
    setup_anndata(test_data, layer="counts")

    device = torch.device("cuda:" + str(args.use_gpu))
    model = TreeSpikeSlab(test_data, tree_depth)
    print("loading model state_dict")
    model.load_state_dict(torch.load(args.model_file))
    model.module.to(device)

    print("getting latent representation of test data")
    test_theta = model.get_latent_representation(test_data, deterministic=True, output_softmax_z=False)

    print("---Saving topic proportions---\n")
    topics_df = pd.DataFrame(test_theta, index=test_data.obs.index, columns=['topic_' + str(j) for j in range(test_theta.shape[1])])
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    topics_df.to_csv(os.path.join(args.out_dir, "topics.csv"))

if __name__ == "__main__":
    main()
