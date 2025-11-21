import sys
import os
import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

# GEARS import
from gears import PertData, GEARS


# ----------------------------
# Argument parser
# ----------------------------
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


# Load ADATA
adata = sc.read(path_to_adata)
print("Loaded ADATA:", adata)


# GEARS expects a 'condition' column
if "condition" not in adata.obs:
    raise RuntimeError("ERROR: adata.obs must contain 'condition' column for GEARS.")

if "split" not in adata.obs:
    raise RuntimeError("ERROR: adata.obs must contain 'split' column (train/val/test).")


# Prepare PertData and model
pd_obj = PertData(adata)
pd_obj.prepare_split(split_col="split", perturbation_key="condition")

model = GEARS(pd_obj)
model.model_initialize()

# Create model directory
os.makedirs(path_to_model, exist_ok=True)

# Train GEARS
print("Starting GEARS training...")
model.train(
    path_to_save=path_to_model,
    lr=1e-4,
    batch_size=512,
    max_epochs=50,
    verbose=True
)
print("Finished GEARS training.")


# Predict on TEST set
test_conditions = adata.obs.loc[adata.obs["split"] == "test", "condition"].unique()
print("Test conditions:", test_conditions)

pred_dict = model.predict(test_conditions, split="test", return_type="mean")


# Write predictions into ADATA layers
n_cells, n_genes = adata.n_obs, adata.n_vars
adata.layers["pred_gears"] = np.full((n_cells, n_genes), np.nan, dtype=float)

for pert, pred in pred_dict.items():
    mask = adata.obs["condition"] == pert
    adata.layers["pred_gears"][mask] = pred

# Do NOT include condition if original file didn't have it
# (your preprocessing script added it — so keep it)
# If you want to remove it like scLambda:
# del adata.obs["condition"]


# Export final h5ad
output_path = f"{path_to_results}/{dataset_name}_pred_gears_{trial_number}.h5ad"
print("Exporting to:", output_path)
sc.write(output_path, adata)

print("Done.")