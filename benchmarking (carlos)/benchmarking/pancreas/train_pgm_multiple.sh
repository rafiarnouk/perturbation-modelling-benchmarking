#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_%j
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --partition=beaver
#SBATCH --output=/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/benchmarking/pancreas/slurm_out/pgm_multiple.out

source ~/.bashrc
mamba activate pgm

wandb login 3d7526d2fd6bc806aaef3b57ff773069e162227d

cd "/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/train_scripts/"

seeds=(0 1 2 3 4)
for seed in "${seeds[@]}"; do
    python train_pgm.py \
        --path_to_adata="/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/pancreas.h5ad" \
        --path_to_model="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
        --model_name="seed_${seed}" \
        --batch_cols="fake_batch" \
        --num_latents=2 \
        --is_last_z_informed=False \
        --gene_subsets="cell_cycle" \
        --shared_decoder=False \
        --decoder_layer_sizes 128 256 \
        --num_epochs=2500 \
        --is_zk_nograd=False \
        --track_wandb=True \
        --save=False \
        --seed=$seed
done
wait