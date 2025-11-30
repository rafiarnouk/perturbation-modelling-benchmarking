# given adata object with splits, make pred_baseline_mean layer
# should take average gene expression of all cells not in test (in train or val)
# and set preds_baseline to this in all test cells
import argparse
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np

# example usage: 
"""
python baseline_predictor.py \
  --data_path ../data/preprocessed/adamson_preprocessed_0.h5ad \
  --write_path ../data/predictions/adamson_pred_baseline_0.h5ad
"""

def main():
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset")
    parser.add_argument("--data_path", type=str, help="Path to adata object (object should have assigned splits)")
    parser.add_argument("--write_path", type=str, help="Path to write to")
    args = parser.parse_args()
    data_path = args.data_path
    write_path = args.write_path

    adata = sc.read(data_path)
    predict_mean(adata)

    # NOTE if dealing with larger datasets might have to run on arc so update to export to scratch instead
    print(f"Saving to {write_path}")
    sc.write(write_path, adata)

def predict_mean(adata):
    mask_not_test = adata.obs["split"].isin(["train", "val"])
    adata_not_test = adata[mask_not_test]
    mean_gene_expression = adata_not_test.X.mean(axis=0)
    print("Mean gene expression:", mean_gene_expression)

    # make preds_baseline_mean layer
    n_cells, n_genes = adata.n_obs, adata.n_vars
    adata.layers["pred_baseline_mean"] = np.full((n_cells, n_genes), np.nan, dtype=float)
    mask_test = adata.obs["split"] == "test"
    adata.layers["pred_baseline_mean"][mask_test] = mean_gene_expression
    

if __name__ == "__main__":
    main()