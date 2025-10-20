import sys
sys.dont_write_bytecode = True

sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/")
from parse import str2bool

sys.path.append("/ubc/cs/research/beaver/projects/carlos/hyperbolic/models/PGM/")
from PGMadd.vae import PGMadd

import json
import wandb
import numpy as np
import pandas as pd
import anndata as ad

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_adata", type=str, required=True, help="Path to the AnnData file")
parser.add_argument("--path_to_model", type=str, required=True, help="Path to save full model")
parser.add_argument("--model_name", type=str, required=True, help="Model name / spaces to use")

# Batch
parser.add_argument("--batch_cols", type=str, nargs="+", default="fake_batch", help="Batch covariates for encoder")
parser.add_argument("--zbatch_cols", type=str, nargs="+", default=None, help="Batch covariates for decoder")

# Latent
parser.add_argument("--num_latents", type=int, required=True, default=2, help="Number of latent variables to infer, i.e. number of hyperbolic spaces")
parser.add_argument("--gaussians_per_latent", type=int, nargs="+", default=[1, 1], help="Number of Gaussians per latent dimension")
parser.add_argument("--is_last_z_informed", type=str2bool, nargs="?", const=True, default=False, help="Whether last latent dimension is informed by all other latents")
parser.add_argument("--gene_subsets", type=str, required=True, help="Prior knowledge (set of markers) to use to mask latents")

# Decoder
parser.add_argument("--shared_decoder", type=str2bool, nargs="?", const=True, default=False, help="Whether to use shared decoder across latents")
parser.add_argument("--decoder_layer_sizes", type=int, nargs="+", default=[128], help="Hidden layer sizes for decoder")
parser.add_argument("--projection_size", type=int, default=None, help="Projection size for first layer of shared decoder")

# Training
parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--is_zk_nograd", type=str2bool, nargs="?", const=True, default=False, help="Whether to disable gradients for zk")
parser.add_argument("--start_zk_nograd", type=int, default=None, help="Epoch to start zk no-grad")
parser.add_argument("--end_zk_nograd", type=int, default=None, help="Epoch to end zk no-grad")

# Misc
parser.add_argument("--track_wandb", type=str2bool, nargs="?", const=True, default=False, help="Whether to track model run on WANDB or not")
parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
parser.add_argument("--save", type=str2bool, nargs="?", const=True, default=False, help="Whether to save the trained model or not")
parser.add_argument("--seed", type=int, default=2025, help="Random seed for model run")
args = parser.parse_args()

dataset = args.path_to_adata.split("/")[-2]
if args.track_wandb:
    wandb.init(
        project="hyperbolic",
        name=dataset + "_" + args.model_name,
        dir="/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/wandb_tmp/",
        config=vars(args)
    )

##############################
### Load data & prior file ###
##############################
adata = ad.read_h5ad(args.path_to_adata)

if args.gene_subsets == "cell_cycle":
    CU_cellcycle = pd.read_csv("/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/celluntangler_mouse_cell_cycle_genes.tsv", sep="\t")
    prior = [CU_cellcycle[CU_cellcycle["in_adata"]]["gene"].tolist()]
elif args.gene_subsets == "ov_cell_type":
    OV_celltype = pd.read_csv("/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/ovarian/ovarian_celltype_markers.tsv", sep="\t")
    prior = []
    for ct in OV_celltype["cell_type"].unique():
        ct_gene_names = OV_celltype[OV_celltype["cell_type"] == ct]["gene"]
        ct_gene_ids = adata.var[adata.var["feature_name"].isin(ct_gene_names)].index.tolist()
        prior.append(ct_gene_ids)

####################
### Set up model ###
####################
model = PGMadd(
    adata,
    batch_cols=args.batch_cols,
    zbatch_cols=[[col] for col in args.zbatch_cols], # args.zbatch_cols
    num_latents=args.num_latents,
    gaussians_per_latent=args.gaussians_per_latent,
    is_last_z_informed=args.is_last_z_informed,
    decoder_layer_sizes=args.decoder_layer_sizes,
    projection_size=args.projection_size,
    shared_decoder=args.shared_decoder,
    gene_subsets=prior,

    device=args.device,
    track_wandb=args.track_wandb,
    seed=args.seed
)
print(model)

###############################
### Train & save embeddings ###
###############################
model.train_model(
    mb_size=128,
    lr=1e-4,
    num_epochs=args.num_epochs,
    is_zk_nograd=args.is_zk_nograd,
    start_zk_nograd=args.start_zk_nograd,
    end_zk_nograd=args.end_zk_nograd
)

save_str = f"{args.path_to_model}pgm_{args.model_name}"

zmu, zvar = model.get_latent_embeddings()
for k in range(args.num_latents):
    np.savetxt(
        f"{save_str}_z{k+1}.tsv",
        np.concatenate((zmu[:,[k]], zvar[:,[k]]), axis=1),
        delimiter="\t"
    )

# Save model arguments
with open(f"{save_str}_config.json", "w") as f:
    json.dump(vars(args), f, indent=2)