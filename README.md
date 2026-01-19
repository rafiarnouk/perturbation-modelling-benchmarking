A repository to store utility functions and any other general code related to the benchmarking of ML models made for the task of perturbation modelling.

Datasets can be found at https://projects.sanderlab.org/scperturb/datavzrd/scPerturb_vzrd_v1/dataset_info/index_1.html
* `norman.h5ad` is NormanWeissman2019 with 237 perturbations
* `adamson.h5ad` is AdamsonWeissman2016 with 20 perturbations
* `replogle.h5ad` is ReplogleWeissman2022 (rpe1) with 2394 perturbations

### How to use the pipeline

1. **Upload data**: upload raw adata object to `data/` directory
2. **Preprocess and split data**: use the `preprocessor.py` script to save a new preprocessed adata object with splits assigned to the `preprocessed/` directory for each trial/seed
    - example usage:
    
    ```bash
    python preprocessor.py adamson --split --seeds 0 1 2
    ```
    
3. **Train model**: write a training script for the model you are benchmarking that takes each preprocessed adata object, trains your model, and saves an adata object to the `predictions/` folder with a new layer named `pred_{your_model}` that has predicted gene expressions for cells in the test split
    - example (can refer to `framework/training/train_sclambda.py` for training script example):
    
    ```bash
    seeds=(0 1 2)
    
    for seed in "${seeds[@]}"; do
        python train_sclambda.py \
            --path_to_adata "/.../perturbation-modelling/data/adamson_preprocessed_${seed}.h5ad" \
            --path_to_embeddings "/.../perturbation-modelling/data/GPT_3_5_gene_embeddings_3-large.pickle" \
            --path_to_model "/.../models_Normal_split0" \
            --path_to_results "/.../results" \
            --dataset_name "adamson" \
            --trial_number $seed
    done
    ```
    
4. **Calculate metrics**: use the `metric_calculator.py` script, passing paths to adata objects with prediction layers, to calculate metrics
    - example usage:
    
    ```bash
    python metric_calculator.py \
      --preds_files adamson_pred_sclambda_0 adamson_pred_sclambda_1 adamson_pred_sclambda_2 \
      --metrics pearson \
      --preds_layer_name pred_sclambda
    ```
    

Additionally, you can create baselines using the `baseline_predictor.py` script to use for comparison with the other models. Here is an example of how to use this script:

```bash
python baseline_predictor.py \
  --data_path ../data/preprocessed/adamson_preprocessed_0.h5ad \
  --write_path ../data/predictions/adamson_pred_baseline_0.h5ad
```
