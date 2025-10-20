source ~/.bashrc
mamba activate pgm

cd ..

python eval_clusterings.py \
    --path_to_adata="/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/pancreas.h5ad" \
    --path_to_embeddings="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
    --path_to_savedf="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
    --truth_obs_key="clusters_fine"

python eval_clusterings.py \
    --path_to_adata="/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/pancreas.h5ad" \
    --path_to_embeddings="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
    --path_to_savedf="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
    --truth_obs_key="ccPhase"

# Only cycling cells
python eval_clusterings.py \
    --path_to_adata="/ubc/cs/research/beaver/projects/carlos/hyperbolic/data/pancreas/pancreas.h5ad" \
    --path_to_embeddings="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
    --path_to_savedf="/ubc/cs/research/beaver/projects/carlos/hyperbolic/results/benchmarking/pancreas/pgm_cu_5runs/" \
    --truth_obs_key="ccPhase" \
    --subset_obs_key="clusters_fine" \
    --subset_obs_vals "Ductal" "Ngn3 low EP"
