import scanpy as sc
import argparse
import re
import random
import numpy as np
from utils import assign_splits, assign_splits_proportional, get_perturbed_genes_map
from pathlib import Path


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset")
    parser.add_argument("dataset", type=str, help="Name of the dataset (e.g. 'adamson')")
    parser.add_argument("--split", action="store_true", help="If --split included, export train/test/val splits")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, defaults to zero")
    args = parser.parse_args()
    dataset = args.dataset
    should_split = args.split
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
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
    if "log1p" in adata.uns:  # remove adata.uns["log1p"] = {"base": None} since this causes problem when reading in earlier anndata versions
        del adata.uns["log1p"]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, layer="counts")
    adata = adata[:,adata.var["highly_variable"]]

    # remove irrelevant rows
    adata = adata[~adata.obs["perturbation"].isna() & (adata.obs["perturbation"] != "*")].copy()

    # remove combinatorial perturbations (keep control perturbations if they are still in the data)
    pert_map = get_perturbed_genes_map(adata)
    single_perts = [pert for pert, genes in pert_map.items() if len(genes) == 1 or re.search(r"ctrl|control", pert, flags=re.IGNORECASE)]
    adata = adata[adata.obs["perturbation"].isin(single_perts)].copy()

    # assign train/val/test splits
    if should_split:
        split = {"train": 0.7, "val": 0.15, "test": 0.15}
        assign_splits_proportional(adata, split)

    # print(adata.obs["perturbation"].value_counts())
    # print(adata.obs["split"].value_counts())
    print("perts in train", list(adata.obs.loc[adata.obs["split"] == "train", "perturbation"].unique()))
    print("perts in test", list(adata.obs.loc[adata.obs["split"] == "test", "perturbation"].unique()))
    print("perts in val", list(adata.obs.loc[adata.obs["split"] == "val", "perturbation"].unique()))

    # save updated adata object
    write_path = f"{data_path}/preprocessed"
    print("Saving data")
    sc.write(f"{write_path}/{dataset}_preprocessed_{seed}.h5ad", adata)

if __name__ == "__main__":
    main()