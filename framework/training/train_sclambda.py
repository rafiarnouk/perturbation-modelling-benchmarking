# not meant to be ran inside the framework repo
# should be ran inside sclambda repo so the model can be accessed, using virtual env provided
# TODO generalize away from adamson dataset

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

gene_embeddings = pd.read_pickle('/arc/project/st-jiaruid-1/rarnou01/perturbation-modelling/data/GPT_3_5_gene_embeddings_3-large.pickle')
adata = sc.read('/arc/project/st-jiaruid-1/rarnou01/perturbation-modelling/data/adamson_preprocessed.h5ad')
print("done reading data")

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

model = sclambda.model.Model(adata,
                             gene_embeddings,
                             model_path = "/scratch/st-jiaruid-1/rarnou01/models_Normal_split0",
                             multi_gene = False,
                             training_epochs = 10)

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
results_path = "/scratch/st-jiaruid-1/rarnou01/results"
sc.write(f"{results_path}/adamson_pred_sclambda.h5ad", adata)
