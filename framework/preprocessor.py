import scanpy as sc
import argparse
from utils import assign_splits
from pathlib import Path
import scanpy as sc
import re

def main():
    # parse command-line argument
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset")
    parser.add_argument("dataset", type=str, help="Name of the dataset (e.g. 'adamson')")
    parser.add_argument("--split", action="store_true", help="If --split included, export train/test/val splits")
    args = parser.parse_args()
    dataset = args.dataset
    should_split = args.split
    data_path = str(Path(__file__).parent.parent / "data")

    # read dataset
    print(f"Reading dataset {dataset}")
    adata = sc.read(f"{data_path}/{dataset}.h5ad")

    # feature selection and normalization
    print("Preprocessing and cleaning data")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, layer="counts")
    adata = adata[:,adata.var["highly_variable"]]

    # remove irrelevant rows
    adata = adata[~adata.obs["perturbation"].isna() & (adata.obs["perturbation"] != "*")].copy()

    # remove control perturbations
    adata = adata[~adata.obs["perturbation"].str.contains(r"ctrl|control", flags=re.IGNORECASE, na=False)].copy()

    # assign train/val/test splits
    if should_split:
        split = {"train": 0.7, "val": 0.15, "test": 0.15}
        assign_splits(adata, split)

    # save updated adata object
    write_path = f"{data_path}/preprocessed"
    print("Saving data")
    sc.write(f"{write_path}/{dataset}_preprocessed.h5ad", adata)

if __name__ == "__main__":
    main()