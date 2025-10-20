import sys
sys.dont_write_bytecode = True

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import os
import json
sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/")
from parse import str2bool

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_adata", type=str, required=True, help="Path to the AnnData file")
parser.add_argument("--path_to_model", type=str, required=True, help="Path to save full model")
parser.add_argument("--name", type=str, required=True, help="Name of experiment / run")
parser.add_argument("--model_components", type=str, required=True, help="Components / spaces to use")
parser.add_argument("--z2_no_grad", type=str2bool, nargs="?", const=True, default=False, help="Whether to stop gradient for z2 or not")
parser.add_argument("--prior", type=str, required=True, help="Prior knowledge (set of markers) to use")
parser.add_argument("--seed", type=int, required=True, help="Seed for model run")
args = parser.parse_args()

sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/models/CellUntangler/")
from src.celluntangler.models.get_config import get_config
from src.data.umi_data import UMIVaeDataset
from src.celluntangler import utils
from src.celluntangler.models import Trainer
from src.celluntangler.models.nb_vae import NBVAE
from src.visualization.helpers import split_embeddings, lorentz_to_poincare

sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/")
from parse import str2bool

# Set parser arguments
path_to_adata = args.path_to_adata
path_to_model = args.path_to_model
name = args.name
model_components = args.model_components
z2_no_grad = args.z2_no_grad
prior = args.prior
seed = args.seed

print(f"Running CellUntangler  model {args.name} with spaces {args.model_components}")
# Rest of the code is based off of CellUntangler's tutorial
##############################
### Initialize config file ###
##############################
print(f"\nLoading config file")
config = get_config()
config.model_name = model_components
config.seed = seed
config.init = "custom"

config.component_subspaces = {}

torch.set_default_dtype(torch.float64)
config.device = torch.device("cpu")

if z2_no_grad:
    config.use_z2_no_grad = True

print(config)
#################
### Load data ###
#################
print(f"\nLoading data")

# X/y/batch
adata = ad.read_h5ad(path_to_adata)
x = adata.layers["counts"].todense().astype(np.double)
batch = (np.zeros((x.shape[0], 1)) * -1).astype(np.int64)
y = batch

# Dataset / dataloader
dataset = UMIVaeDataset(batch_size=config.batch_size, in_dim=x.shape[1])
train_loader = dataset.create_loaders(x, y, seed=config.seed)

# Masking
if prior == "cell_cycle":
    prior_genes = pd.read_csv("/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/celluntangler_mouse_cell_cycle_genes.tsv", sep="\t")
    prior_genes = prior_genes[prior_genes["in_adata"]]["gene"].tolist()

mask_z1 = np.zeros(adata.n_vars)
mask_z1[adata.var.index.isin(prior_genes)] = 1

mask_z2 = np.ones(adata.n_vars)
mask_z2[adata.var.index.isin(prior_genes)] = 0

mask = torch.tensor([mask_z1, mask_z2])

###################################
### Setting up & training model ###
###################################
print(f"\n Setting up and training model")

if config.seed:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    np.random.default_rng(config.seed)

visualize_information=None
components = utils.parse_components(config.model_name, config.fixed_curvature)
model = NBVAE(h_dim=config.h_dim,
              components=components,
              mask=mask,
              dataset=dataset,
              config=config,
              component_subspaces=None
        ).to(config.device)

print(model)
trainer = Trainer(model)

optimizer = trainer.build_optimizer(learning_rate=config.learning_rate,
                                        fixed_curvature=config.fixed_curvature,
                                        use_adamw=config.use_adamw,
                                        weight_decay=config.weight_decay)

betas = utils.linear_betas(config.start,
                           config.end,
                           end_epoch=config.end_epoch,
                           epochs=config.epochs)

trainer.train_epochs(optimizer=optimizer,
                       train_data=train_loader,
                       betas=betas,
                       likelihood_n=0,
                       max_epochs=config.max_epochs,
                       visualize_information=visualize_information)

#######################
### Save embeddings ###
#######################
print(f"\nSaving embeddings")

save_str = f"{args.path_to_model}cu_{''.join(args.model_components.split(','))}_{args.name}"
embeddings_save_path = path_to_model

a = trainer.model(torch.log1p(torch.tensor(x, device=config.device)), torch.tensor(y, device=config.device))
embeddings = a[4].detach().to(torch.device("cpu")).numpy()

component_embeddings = split_embeddings(model_components, embeddings)
for k, c in enumerate(component_embeddings):
    np.savetxt(
        f"{save_str}_{c}_z{k+1}.tsv", 
        lorentz_to_poincare(component_embeddings[c], curvature=-2), 
        delimiter="\t"
    )

# Save model's arguments
with open(f"{save_str}_config.json", "w") as f:
    json.dump(vars(args), f, indent=2)
