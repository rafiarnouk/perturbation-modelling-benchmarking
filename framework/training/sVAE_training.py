import argparse
import logging
import os
import numpy as np
import torch
import scanpy as sc


from svae import SpikeSlabVAE,  sVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser("sVAE benchmark experiment")

    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="simulation")
    parser.add_argument("--n_latent", type=int, default=15)
    parser.add_argument("--n_cells_per_chem", type=int, default=250)
    parser.add_argument("--n_chem", type=int, default=100)
    parser.add_argument("--n_genes", type=int, default=100)
    parser.add_argument("--n_epoch", type=int, default=300)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--sparse_penalty", type=float, default=0)
    parser.add_argument("--method", type=str, default="SpikeSlabVAE")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--trial_number", type=int, required=True)
    parser.add_argument("--path_to_results", type=str, required=True)

    args = parser.parse_args()

    path_to_adata = args.dataset
    trial_number = args.trial_number
    path_to_results = args.path_to_results

    # set up seeds ############################################################

    torch.backends.cudnn.benchmark = True

    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = f"svae_lat{args.n_latent}mth_{args.method}_sp{args.sparse_penalty}"

    # Read the data
    adata = sc.read(path_to_adata)
    print("Finished reading data")

    # split anndata in train / test
    if args.method != "SpikeSlabVAE":
        sVAE.setup_anndata(adata, labels_key="perturbation")
        model = sVAE(adata, n_latent=args.n_latent, n_layers=args.n_layers)

    if args.method == "SpikeSlabVAE":
        if args.sparse_penalty == 0:
            args.sparse_penalty = 1
        SpikeSlabVAE.setup_anndata(adata, labels_key="perturbation")
        model = SpikeSlabVAE(adata, n_latent=args.n_latent, n_layers=args.n_layers)

    # train or load model #####################################################

    adata_train = adata[adata.obs["split"] == "train"].copy()
    adata_val   = adata[adata.obs["split"] == "val"].copy()
    adata_test  = adata[adata.obs["split"] == "test"].copy()


    chem_prior = (
        args.method == "sVAE" or args.method == "iVAE" or args.method == "SpikeSlabVAE"
    )
    # hack: focus on train only, but leave params for all chemicals
    model.adata = adata_train
    model.module.use_chem_prior = chem_prior
    model.module.sparse_mask_penalty = args.sparse_penalty
    model.train(
        max_epochs=args.n_epoch,
        check_val_every_n_epoch=1,
        early_stopping=True,
        plan_kwargs={
            "n_epochs_kl_warmup": 50,
        }
    )

    elbo_train = model.get_elbo(adata_train, agg=True)
    elbo_val = model.get_elbo(adata_val, agg=True)
    model.save(save_dir, overwrite=True)

    # obtain latents ##########################################################

    latents = model.get_latent_representation(adata_train)

    # compute predictions
    pert_test = adata.obs.loc[adata.obs["split"] == "test", "condition"].unique()
    res = model.predict(pert_test, return_type = "mean")

    # put predictions in adata object
    n_cells, n_genes = adata.n_obs, adata.n_vars
    adata.layers["pred_sVAE"] = np.full((n_cells, n_genes), np.nan, dtype=float)

    for pert, pred in res.items():
        mask = adata.obs["condition"] == pert
        adata.layers["pred_sVAE"][mask] = pred
    
    # export h5ad
    export_path = f"{path_to_results}/adamson_pred_sVAE_{trial_number}.h5ad"
    print(f"Exporting to {export_path}")
    sc.write(export_path, adata)

    