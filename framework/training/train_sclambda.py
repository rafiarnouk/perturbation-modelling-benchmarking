import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sclambda
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import pickle
import re
import matplotlib.pyplot as plt
from sklearn import preprocessing

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_adata", type=str, required=True)
parser.add_argument("--path_to_embeddings", type=str, required=True)
parser.add_argument("--path_to_model", type=str, required=True)
parser.add_argument("--path_to_results", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--trial_number", type=int, required=True)
parser.add_argument("--small_run", action="store_true", required=False)
args = parser.parse_args()

path_to_adata = args.path_to_adata
path_to_embeddings = args.path_to_embeddings
path_to_model = args.path_to_model
path_to_results = args.path_to_results
dataset_name = args.dataset_name
trial_number = args.trial_number
small_run = args.small_run

# example usage: 
"""
python train_sclambda.py \
  --path_to_adata /arc/project/st-jiaruid-1/rarnou01/perturbation-modelling/data/adamson_preprocessed.h5ad \
  --path_to_embeddings /arc/project/st-jiaruid-1/rarnou01/perturbation-modelling/data/GPT_3_5_gene_embeddings_3-large.pickle \
  --path_to_model /scratch/st-jiaruid-1/rarnou01/models_Normal_split0 \
  --path_to_results /scratch/st-jiaruid-1/rarnou01/results \
  --dataset_name adamson \
  --trial_number 0 \
  --small_run
"""

# read data
gene_embeddings = pd.read_pickle(path_to_embeddings)
adata = sc.read(path_to_adata)
print("done reading data")

def prepare_adamson(adata):
    # need to have adata.obs["condition"] without extra metadata in pert name
    adata.obs["condition"] = adata.obs["perturbation"].apply(
        lambda pert: "ctrl" if re.search(r"ctrl|control", pert, flags=re.IGNORECASE)
        else pert.split("_")[0].replace("(mod)", "")
    )

    # check which perts we have embeddings for
    # for pert in adata.obs["condition"].unique():
    #     if pert not in gene_embeddings:
    #         print("No embedding for", pert)
    #     else:
    #         print("Has embedding for", pert)

    # need gene embedding aliases for (Gal4-4, C7orf26, PERK, IRE1)
    # No embedding for Gal4-4: LGALS4
    # No embedding for C7orf26: none found
    # No embedding for PERK: EIF2AK3
    # No embedding for IRE1: ERN1

    # remove C7orf26 since genePT does not have any gene embeddings for this gene (or aliases)
    adata = adata[adata.obs["condition"] != "C7orf26"].copy()

    # update some gene IDs with aliases found from www.ncbi.nlm.nih.gov/gene
    gene_alias_map = {
        "Gal4-4": "LGALS4",
        "PERK": "EIF2AK3",
        "IRE1": "ERN1"
    }
    adata.obs["condition"] = adata.obs["condition"].map(
        lambda x: gene_alias_map.get(x, x)
    )
    return adata

if dataset_name == "adamson":
    adata = prepare_adamson(adata)
else:
    raise RuntimeError("no prep method for dataset provided")

# if we are doing a small test run, only run for 10 epochs (model breaks if epochs < 10)
small_run_args = {"training_epochs": 10} if small_run else {}
model = sclambda.model.Model(adata,
                             gene_embeddings,
                             model_path = path_to_model,
                             multi_gene = False,
                             **small_run_args)
model.train()
print("Exited model training")

# compute predictions
pert_test = adata.obs.loc[adata.obs["split"] == "test", "condition"].unique()
res = model.predict(pert_test, return_type = "mean")

# put predictions in adata object
n_cells, n_genes = adata.n_obs, adata.n_vars
adata.layers["pred_sclambda"] = np.full((n_cells, n_genes), np.nan, dtype=float)
for pert, pred in res.items():
    mask = adata.obs["condition"] == pert
    adata.layers["pred_sclambda"][mask] = pred

# want output adata object to be same format as original, so remove condition column
del adata.obs["condition"]

# export h5ad
export_path = f"{path_to_results}/{dataset_name}_pred_sclambda_{trial_number}.h5ad"
print(f"Exporting to {export_path}")
sc.write(export_path, adata)
