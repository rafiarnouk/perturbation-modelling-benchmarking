import scanpy as sc
import argparse

def main():
    # parse command-line argument
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset")
    parser.add_argument("dataset", type=str, help="Name of the dataset (e.g. 'adamson')")
    args = parser.parse_args()

    dataset = args.dataset

    # read dataset
    print(f"Reading dataset {dataset}")
    adata = sc.read(f"../data/{dataset}.h5ad")

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

    # save updated adata object
    print("Saving to data folder")
    sc.write(f"../data/{dataset}_preprocessed.h5ad", adata)

if __name__ == "__main__":
    main()