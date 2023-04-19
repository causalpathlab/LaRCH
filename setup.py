import os
from setuptools import setup, find_packages

setup(
    name='LaRCH',
    packages=find_packages(),
    version='0.0.1',
    description="LaRCH package, Latent Representation of Cellular Hierarchies. Tree-structured topic modelling",
    url="https://github.com/causalpathlab/SuSiEVAE",
    entry_points={'console_scripts':
        ['spike_slab = larch.run.spike_slab:main',
        'tree_spike_slab = larch.run.tree_spike_slab:main',
        'tree_susie = larch.run.tree_susie:main',
        'sc_data_sim = larch.run.sim_real:main']
    },
    requires=['pandas', 'torch', 'functools', 'h5py', 'numpy', 'anndata', 'pytorch_lightning', 'pickle', 'scanpy']
)
