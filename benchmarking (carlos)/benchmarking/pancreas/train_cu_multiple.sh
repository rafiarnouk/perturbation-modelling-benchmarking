#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_%j
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=18
#SBATCH --partition=beaver
#SBATCH --output=/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/benchmarking/pancreas/slurm_out/cu_multiple.out

source ~/.bashrc
mamba activate cellunt

export LD_LIBRARY_PATH=""
unset LD_PRELOAD

cd "/ubc/cs/research/beaver/projects/carlos/hyperbolic/code/train_scripts/"

seeds=(0 1 2 3 4)
for seed in "${seeds[@]}"; do
    python train_celluntangler.py \
        --path_to_adata="/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/pancreas.h5ad" \
        --path_to_model="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
        --name="seed_${seed}" \
        --model_components="r2, r2" \
        --z2_no_grad=True \
        --prior="cell_cycle" \
        --seed=$seed
done
wait