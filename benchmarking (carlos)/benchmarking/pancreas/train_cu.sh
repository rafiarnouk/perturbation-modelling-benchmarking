#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=train_%j
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --partition=beaver
#SBATCH --output=/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/benchmarking/pancreas/slurm_out/cu.out

source ~/.bashrc
mamba activate cellunt

export LD_LIBRARY_PATH=""
unset LD_PRELOAD

cd "/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/train_scripts"/

python train_celluntangler.py \
    --path_to_adata="/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/pancreas.h5ad" \
    --path_to_model="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/celluntangler/runs/" \
    --name="z2_no_grad" \
    --model_components="r2, r2" \
    --z2_no_grad=True \
    --prior="cell_cycle" \
    --seed=2025
