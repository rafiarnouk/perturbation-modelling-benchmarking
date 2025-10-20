import sys
sys.dont_write_bytecode = True

# Paths for the user to set
import os
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_adata", type=str, required=True, help="Path to the AnnData file")
parser.add_argument("--path_to_embeddings", type=str, required=True, help="Path to the embeddings directory storing z1,..,zK")
parser.add_argument("--path_to_savedf", type=str, required=True, help="Path to save the results TSV")
parser.add_argument("--truth_obs_key", type=str, required=True, help="Column on adata.obs to use as ground truth")
parser.add_argument("--subset_obs_key", type=str, default=None, help="Subset the adata using adata.obs[obs_key]")
parser.add_argument("--subset_obs_vals", type=str, nargs="+", default=None, help="Subset the adata by values in adata.obs[obs_key]")
args = parser.parse_args()

import torch
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/benchmarking/")
from metrics import get_latent_clusterings
from metrics import eval_clusterings

sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/models/PGM/")
from PGMadd import PGMadd

path_to_adata = args.path_to_adata
path_to_embeddings = args.path_to_embeddings
path_to_savedf = args.path_to_savedf
truth_obs_key = args.truth_obs_key
subset_obs_key = args.subset_obs_key
subset_obs_vals = args.subset_obs_vals

#################
### Load data ###
#################
# If we have to subset the adata, then do so here
adata = ad.read_h5ad(path_to_adata)
if subset_obs_key and subset_obs_vals:
    subset_mask = adata.obs[subset_obs_key].isin(subset_obs_vals)
    adata = adata[subset_mask]
else:
    subset_mask = None

######################
### Set up the path ###
######################
embeddings_dir = Path(path_to_embeddings)
embedding_files = list(embeddings_dir.glob("*.tsv"))

###############################
### Iterate over embeddings ###
###############################
res_df_rows = []
for emb_file in tqdm(embedding_files):
    file_name = emb_file.stem.split("_")

    model_name = file_name[0]
    eval_obsm_key = file_name[-1]
    rest_name = "_".join(file_name[1:-1])
    print(f"\nEvaluating {model_name}, latent {eval_obsm_key}")

    # 1. Load embeddings & store onto AnnData
    z = np.loadtxt(emb_file, delimiter="\t")
    if subset_mask is not None:
        z = z[subset_mask]
    adata.obsm[eval_obsm_key] = z

    # If we have to subset cells (e.g. for only cycling cells, do that here)
    if (subset_obs_key is not None) and (subset_obs_vals is not None):
        adata = adata[adata.obs[subset_obs_key].isin(subset_obs_vals)]

    # 2. Cluster the embeddings
    adata = get_latent_clusterings(
        adata,
        latent_obsm_key=eval_obsm_key,
        kmeans_K=adata.obs[truth_obs_key].nunique(),
        seed=2025
    )

    # 4. Evaluate the clusterings
    df = eval_clusterings(
        adata,
        cluster_obs_key=[f"{eval_obsm_key}_kmeans", f"{eval_obsm_key}_leiden", f"{eval_obsm_key}_louvain"],
        truth_obs_key=truth_obs_key
    )

    df["model"] = model_name
    df["latent"] = eval_obsm_key
    df["iter"] = rest_name
    res_df_rows.append(df)

# Save the benchmark
save_str = path_to_savedf + "eval_" + truth_obs_key
if (subset_obs_key is not None) and (subset_obs_vals is not None):
    save_str += "_subset_" + "".join(subset_obs_vals)

res_df = pd.concat(res_df_rows, ignore_index=False)
res_df = res_df.sort_index()
# res_df = res_df.sort_values(by="name")
res_df.to_csv(save_str, sep="\t")