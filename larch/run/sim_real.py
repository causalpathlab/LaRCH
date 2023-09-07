import os
import argparse
from larch.util.simdata_util import sim_real

def main():
    parser = argparse.ArgumentParser(description='Data simulation parameters')
    parser.add_argument('--N', type=int, help='Number of cells', default=5000)
    parser.add_argument('--noise', type=int, help='Noise parameter', default=50)
    parser.add_argument('--seed', type=int, help='seed', default=123)
    parser.add_argument('--bulk_file', help='filepath to bulk expression file', default='data/mean_tpm_merged.csv')
    parser.add_argument('--out_dir', help='output directory for simulated data', default='data')

    args = parser.parse_args()
    print(args)

    N = args.N
    noise = args.noise
    seed = args.seed
    bulk_file = args.bulk_file
    outfile = os.path.join(args.out_dir, f"sim_real_N{N}_noise{noise}_seed{seed}.h5ad")

    if os.path.exists(outfile):
        print("Simulated data already exists, skipping simulation")        

    else:
        sim_real(N, bulk_file, outfile, noise=noise, seed=seed)

if __name__ == "__main__":
    main()
