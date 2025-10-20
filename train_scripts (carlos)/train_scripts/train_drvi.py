# Script based on Tutorials notebook from DRVI readthedocs
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import anndata as ad

import drvi
from drvi.model import DRVI

import sys
sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/")
from parse import str2bool

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_adata", type=str, required=True)
parser.add_argument("--path_to_model", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--latent_dim", type=str, required=True)
parser.add_argument("--batch_cols", type=str, nargs="+", default="fake_batch")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--save", type=str2bool, nargs="?", const=True, default=False)
args = parser.parse_args()

path_to_adata = args.path_to_adata
path_to_model = args.path_to_model
name = args.name
latent_dim = args.latent_dim
batch_cols = args.batch_cols
seed = args.seed
save = args.save

#################
### Load data ###
#################
adata = ad.read_h5ad(path_to_adata)
adata.obs["fake_batch"] = ["fake"] * len(adata)
print(adata)

##################
### Setup DRVI ###
##################
torch.manual_seed(seed)
np.random.seed(seed)
np.random.default_rng(seed)

DRVI.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=[batch_cols],
    is_count_data=True
)

model = DRVI(
    adata,
    categorical_covariates=[batch_cols],
    n_latent=latent_dim,
    encoder_dims=[128, 128],
    decoder_dims=[128, 128]
)
print(model)

##################
### Train DRVI ###
##################
model.train(
    max_epochs=400,
    early_stopping=False
)

###############################
### Save embeddings / model ###
###############################
if save:
    model.save(f"{path_to_model}drvi_{name}", overwrite=True)

# Save each latent representation
latents = model.get_latent_representation()
for k in latents.shape[1]:
    zk = latents[:, [k]]
    np.savetxt(f"{path_to_model}drvi_{name}_z{k+1}.tsv")
