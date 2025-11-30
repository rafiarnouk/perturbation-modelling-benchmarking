#!/bin/bash
#SBATCH --time=20:00:00           
#SBATCH --account=st-jiaruid-1      
#SBATCH --nodes=1                 
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=4	      
#SBATCH --mem=100G                 
#SBATCH --job-name=sVAE-adamson      
#SBATCH -e %j-sVAE-adamson.err          
#SBATCH -o %j-sVAE-adamson.out           
#SBATCH --mail-user=gkim12@student.ubc.ca
#SBATCH --mail-type=ALL           

source ~/.bashrc
conda activate geneformer

cd "/arc/project/st-jiaruid-1/william/sVAE/sVAE"

export NUMBA_CACHE_DIR=/scratch/st-jiaruid-1/gkim12/numba_cache
mkdir -p $MPLCONFIGDIR $NUMBA_CACHE_DIR

python sVAE_training.py \
        --path_to_adata "/arc/project/st-jiaruid-1/william/sVAE/sVAE/data/preprocessed/adamson_preprocessed.h5ad" \
        --trial_number 1 \
        --path_to_results "/arc/project/st-jiaruid-1/william/sVAE/sVAE/data/results"

done