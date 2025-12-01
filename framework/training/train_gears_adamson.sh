#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=st-jiaruid-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --job-name=GEARS-adamson
#SBATCH -e %j-GEARS-adamson.err
#SBATCH -o %j-GEARS-adamson.out
#SBATCH --mail-user=anshshrm2004@gmail.com
#SBATCH --mail-type=ALL

source ~/.bashrc
module load gcc/9.4.0
module load miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate sc-perturb
module load cuda/12.1 2>/dev/null 

TRIAL=0

DATA_H5AD="/arc/project/st-jiaruid-1/asharm/perturbation-modelling-benchmarking/data/preprocessed/adamson_preprocessed.h5ad"
MODEL_DIR="/scratch/st-jiaruid-1/asharm/models_gears_adamson_trial${TRIAL}"
RESULTS_DIR="/scratch/st-jiaruid-1/asharm/results_gears_adamson"

mkdir -p "$MODEL_DIR" "$RESULTS_DIR"

cd "/arc/project/st-jiaruid-1/asharm/perturbation-modelling-benchmarking" 

echo "Running GEARS training on Adamson..."
python framework/training/train_gears.py \
    --path_to_adata "$DATA_H5AD" \
    --path_to_model "$MODEL_DIR" \
    --path_to_results "$RESULTS_DIR" \
    --dataset_name adamson \
    --trial_number "$TRIAL"

echo "Done."