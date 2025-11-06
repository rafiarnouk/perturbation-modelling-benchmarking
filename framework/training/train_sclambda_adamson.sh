#!/bin/bash
#SBATCH --time=20:00:00           
#SBATCH --account=st-jiaruid-1      
#SBATCH --nodes=1                 
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=4	      
#SBATCH --mem=100G                 
#SBATCH --job-name=sclambda-adamson      
#SBATCH -e %j-sclambda-adamson.err          
#SBATCH -o %j-sclambda-adamson.out           
#SBATCH --mail-user=rafi.arnouk@gmail.com  
#SBATCH --mail-type=ALL           

source ~/.bashrc
mamba activate sclambda

cd "/arc/project/st-jiaruid-1/rarnou01/perturbation-modelling/scLAMBDA"

export NUMBA_CACHE_DIR=/scratch/st-jiaruid-1/rarnou01/numba_cache
mkdir -p $MPLCONFIGDIR $NUMBA_CACHE_DIR

seeds=(0 1 2)

# remember to remove --small_run flag before real run
for seed in "${seeds[@]}"; do
    python train_sclambda.py \
        --path_to_adata "/arc/project/st-jiaruid-1/rarnou01/perturbation-modelling/data/adamson_preprocessed_${seed}.h5ad" \
        --path_to_embeddings "/arc/project/st-jiaruid-1/rarnou01/perturbation-modelling/data/GPT_3_5_gene_embeddings_3-large.pickle" \
        --path_to_model "/scratch/st-jiaruid-1/rarnou01/models_Normal_split0" \
        --path_to_results "/scratch/st-jiaruid-1/rarnou01/results" \
        --dataset_name "adamson" \
        --trial_number $seed
done
