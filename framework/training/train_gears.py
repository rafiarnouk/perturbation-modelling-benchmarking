import sys
import os
import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import pickle

from gears import PertData, GEARS
import gears.gears as gears_gears
import gears.inference as gears_inference
import torch.nn.functional as F
from scipy.stats import pearsonr as orig_pearsonr
import warnings
warnings.filterwarnings(action="ignore")

def loss_all_genes(pred, y, *args, **kwargs):
    return F.mse_loss(pred, y)

gears_gears.loss_fct = loss_all_genes

def safe_pearsonr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return 0.0, 1.0
    return orig_pearsonr(x, y)

gears_inference.pearsonr = safe_pearsonr

def safe_deeper_analysis(adata, test_res):
    return None, None

gears_inference.deeper_analysis = safe_deeper_analysis
gears_gears.deeper_analysis = safe_deeper_analysis
def safe_non_dropout_analysis(adata, test_res):
    return None, None

gears_inference.non_dropout_analysis = safe_non_dropout_analysis
gears_gears.non_dropout_analysis = safe_non_dropout_analysis


#Argument parser

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_adata", type=str, required=True)
parser.add_argument("--path_to_model", type=str, required=True)
parser.add_argument("--path_to_results", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--trial_number", type=int, required=True)
args = parser.parse_args()

path_to_adata = args.path_to_adata
path_to_model = args.path_to_model
path_to_results = args.path_to_results
dataset_name = args.dataset_name
trial_number = args.trial_number

#Load and prepare AnnData

adata = sc.read(path_to_adata)
print("Loaded ADATA:", adata)

if "condition" not in adata.obs:
    if "perturbation" in adata.obs:
        adata.obs["condition"] = adata.obs["perturbation"].copy()
    else:
        raise RuntimeError(
            "ERROR: need either 'condition' or 'perturbation' in adata.obs for GEARS."
        )

if "cell_type" not in adata.obs:
    if "celltype" in adata.obs:
        adata.obs["cell_type"] = adata.obs["celltype"].copy()
    else:
        adata.obs["cell_type"] = "celltype_0"

if "gene_name" not in adata.var:
    adata.var["gene_name"] = adata.var_names

if "split" not in adata.obs:
    raise RuntimeError("adata.obs must contain 'split' column (train/val/test).")

# Base non_zeros_gene_idx on all conditions (good default)
conditions = adata.obs["condition"].unique().tolist()
n_genes = adata.n_vars
mapping = {}

for c in conditions:
    mapping[c] = np.arange(n_genes, dtype=int)

    if "_" in c:
        base = c.split("_")[0]
        if base not in mapping:
            mapping[base] = np.arange(n_genes, dtype=int)

    if "+" in c:
        for part in c.split("+"):
            part = part.strip()
            if part and part not in mapping:
                mapping[part] = np.arange(n_genes, dtype=int)

adata.uns["non_zeros_gene_idx"] = mapping

#GEARS PertData + custom split

gears_root = os.path.join(os.path.dirname(path_to_adata), "gears_cache")
os.makedirs(gears_root, exist_ok=True)

pd_obj = PertData(gears_root)
dataset_id = f"{dataset_name}_trial{trial_number}"
pd_obj.new_data_process(dataset_name=dataset_id, adata=adata, skip_calc_de=True)

splits_dir = os.path.join(gears_root, dataset_id, "splits")
os.makedirs(splits_dir, exist_ok=True)
split_dict_path = os.path.join(splits_dir, f"{dataset_id}_custom_split.pkl")

split_dict = {}
for s in ["train", "val", "test"]:
    conds = (
        adata.obs.loc[adata.obs["split"] == s, "condition"]
        .unique()
        .tolist()
    )
    split_dict[s] = conds

with open(split_dict_path, "wb") as f:
    pickle.dump(split_dict, f)

pd_obj.prepare_split(split="custom", seed=trial_number, split_dict_path=split_dict_path)
pd_obj.get_dataloader(batch_size=512, test_batch_size=512)

#GEARS model

model = GEARS(pd_obj, device="cpu")
model.model_initialize()

os.makedirs(path_to_model, exist_ok=True)

print("Starting")
model.train(epochs=1)
model.save_model(path_to_model)
print("Finished")

#Prediction on test conditions

test_conditions = pd_obj.set2conditions.get("test", [])
print("Raw test conditions from GEARS:", test_conditions)

pred_dict = {}

for cond in test_conditions:
    try:
        out = model.predict([cond]) 
        if cond in out:
            pred_dict[cond] = out[cond]
        else:
            print(f"Warning: GEARS.predict did not return key {cond}, skipping.")
    except ValueError as e:
        print(f"Skipping {cond} due to GEARS error: {e}")

n_cells, n_genes = adata.n_obs, adata.n_vars
adata.layers["pred_gears"] = np.full((n_cells, n_genes), np.nan, dtype=float)

for cond, pred in pred_dict.items():
    mask = adata.obs["condition"] == cond
    adata.layers["pred_gears"][mask] = pred

os.makedirs(path_to_results, exist_ok=True)
output_path = f"{path_to_results}/{dataset_name}_pred_gears_{trial_number}.h5ad"
print("Exporting to:", output_path)
sc.write(output_path, adata)

print("Done.")