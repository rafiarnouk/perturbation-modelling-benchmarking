import sys
sys.dont_write_bytecode = True

sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/")
from parse import str2bool

import scipy
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FastICA

import sys
sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/")
from parse import str2bool

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_adata", type=str, required=True)
parser.add_argument("--path_to_model", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--X_layer", type=str)
parser.add_argument("--n_components", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--save", type=str2bool, nargs="?", const=True, default=False)
args = parser.parse_args()

path_to_adata = args.path_to_adata
path_to_model = args.path_to_model
name = args.name
X_layer = args.X_layer
n_components = args.n_components
seed = args.seed
save = args.save

#################
### Load data ###
#################
adata = ad.read_h5ad(path_to_adata)
print(adata)

###############
### Model ###
###############
# Based on DRVI. By default, use log-normalized counts 
if X_layer is not None:
    X = adata.layers["counts"]
else:
    X = adata.X

if scipy.sparse.issparse(X):
    X = X.astype(np.float32).A

model = Pipeline([
    ("scaling", StandardScaler(with_mean=True, with_std=False)),
    ("ICA", FastICA(
        n_components=n_components, 
        random_state=seed,
        whiten="unit-variance",
        whiten_solver="eigh"
    ))
])

model.fit(X)
latents = model.transform(X)

#######################
### Save embeddings ###
#######################