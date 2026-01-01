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
  --data_files norman_preprocessed_0 norman_preprocessed_1 norman_preprocessed_2 \
  --write_filenames norman_pred_baseline_0 norman_pred_baseline_1 norman_pred_baseline_2 \
  --data_path /scratch/st-jiaruid-1/rarnou01/preprocessed \
  --write_path /scratch/st-jiaruid-1/rarnou01/results
"""

def main():
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset")
    parser.add_argument("--data_files", nargs="*", type=str, help="Files containing adata objects to add baseline preds to")
    parser.add_argument("--write_filenames", nargs="*", type=str, help="Names for output adata object files")
    parser.add_argument("--data_path", type=str, help="Path to adata object (object should have assigned splits)")
    parser.add_argument("--write_path", type=str, help="Path to write to")
    args = parser.parse_args()
    data_files = args.data_files
    write_filenames = args.write_filenames
    data_path = args.data_path
    write_path = args.write_path

    for file, output_filename in zip(data_files, write_filenames):
        adata = sc.read(f"{data_path}/{file}.h5ad")
        predict_mean(adata)
        print(f"Saving to {write_path}/{output_filename}.h5ad")
        sc.write(f"{write_path}/{output_filename}.h5ad", adata)

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