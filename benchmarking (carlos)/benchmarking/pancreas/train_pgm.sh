#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=train_%j
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --partition=beaver
#SBATCH --output=/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/benchmarking/pancreas/slurm_out/pgm.out

source ~/.bashrc
mamba activate pgm

wandb login 3d7526d2fd6bc806aaef3b57ff773069e162227d

cd "/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/train_scripts"

python train_pgm.py \
    --path_to_adata="/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/pancreas.h5ad" \
    --path_to_model="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/runs/" \
    --model_name="uq_2latents_CUcc_z1_nograd" \
    --batch_cols="fake_batch" \
    --num_latents=2 \
    --is_last_z_informed=true \
    --gene_subsets="cell_cycle" \
    --shared_decoder=false \
    --projection_size=10 \
    --decoder_layer_sizes 128 256 \
    --num_epochs=2500 \
    --is_zk_nograd=true \
    --start_zk_nograd=0 \
    --end_zk_nograd=2500 \
    --track_wandb=true \
    --seed=2025

