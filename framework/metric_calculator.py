import argparse
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from utils import compute_degs, is_control, pseudobulk_means

# example usage: 
"""
python metric_calculator.py \
  --preds_files adamson_pred_sclambda_0 adamson_pred_sclambda_1 adamson_pred_sclambda_2 \
  --path_to_preds /scratch/st-jiaruid-1/rarnou01/results \
  --metrics pearson \
  --preds_layer_name pred_sclambda

for baselines:
python metric_calculator.py \
  --preds_files adamson_pred_baseline_0 adamson_pred_baseline_1 adamson_pred_baseline_2 \
  --path_to_preds /scratch/st-jiaruid-1/rarnou01/results \
  --metrics pearson \
  --preds_layer_name pred_baseline_mean
"""

def main():
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset")
    parser.add_argument("--preds_files", nargs="*", type=str, help="Names of the files (should be in data/predictions)")
    parser.add_argument("--path_to_preds", type=str, help="Path to folder containing adata objects with predictions")
    parser.add_argument("--metrics", nargs="*", type=str, default=["pearson"], help="Metrics to calculate")
    parser.add_argument("--preds_layer_name", type=str, default=["pred"], help="Name of the predictions layer in the adata object")
    args = parser.parse_args()
    files = args.preds_files
    path_to_preds = args.path_to_preds
    metrics = args.metrics
    preds_layer_name = args.preds_layer_name

    metric_dict = {metric: [] for metric in metrics}

    for split_number, file in enumerate(files):
        print(f"\nSPLIT {split_number}")
        adata = sc.read(f"{path_to_preds}/{file}.h5ad")
        test_rows = ~np.all(np.isnan(adata.layers[preds_layer_name]), axis=1)
        actual = adata.X.toarray()[test_rows]
        pred = adata.layers[preds_layer_name][test_rows]

        if "pearson" in metrics:
            corr = calculate_pearson(actual, pred)
            print(f"Mean pearson correlation across cells for split {split_number}: {corr:.3f}")
            metric_dict["pearson"].append(corr)
        if "weighted_delta_r2" in metrics:
            r2 = calculate_weighted_delta_r2(adata, preds_layer_name)
            print(f"Mean weighted delta R2 across perts for split {split_number}: {r2:.3f}")
            metric_dict["weighted_delta_r2"].append(r2)

    for metric, results in metric_dict.items():
        print(f"Average {metric} across all splits: {(sum(results) / len(results)):.3f}")

def calculate_pearson(actual, pred):
    correlations = np.array([
        np.corrcoef(actual[i, :], pred[i, :])[0, 1]
        for i in range(actual.shape[0])
    ])

    # histogram code
    # plt.hist(correlations, bins=50)
    # plt.xlabel("Per-cell correlation")
    # plt.ylabel("Number of cells")
    # plt.title("Predicted vs Actual Expression Correlation (per cell)")
    # plt.show()
    return np.nanmean(correlations)

def calculate_weighted_delta_r2(adata, preds_layer_name):
    curr_deg_results = compute_degs(adata)
    names_df_vsrest = pd.DataFrame(curr_deg_results["names"])
    scores_df_vsrest = pd.DataFrame(curr_deg_results["scores"])
    pert_normalized_abs_scores_vsrest = {}
    for pert in scores_df_vsrest.columns:
        if pert == 'control': # Typically no scores for control in vsrest, but good to check
            continue

        abs_scores = np.abs(scores_df_vsrest[pert].values) # Ensure it's a numpy array
        min_val = np.min(abs_scores)
        max_val = np.max(abs_scores)

        if max_val == min_val:
            if max_val == 0: # All scores are 0
                normalized_weights = np.zeros_like(abs_scores)
            else: # All scores are the same non-zero value
                # Squaring ones will still be ones, which is fine.
                normalized_weights = np.ones_like(abs_scores) 
        else:
            normalized_weights = (abs_scores - min_val) / (max_val - min_val)

        # Ensure no NaNs in weights, replace with 0 if any (e.g. if a gene had NaN score originally)
        normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)

        # Make weighting stronger by squaring the normalized weights
        stronger_normalized_weights = np.square(normalized_weights)

        weights = pd.Series(stronger_normalized_weights, index=names_df_vsrest[pert].values, name=pert)
        # Order by the var_names
        weights = weights.reindex(adata.var_names)
        pert_normalized_abs_scores_vsrest[pert] = weights

    perts = adata.obs.loc[adata.obs["split"] == "test", "perturbation"].unique().tolist()
    r2_per_pert_dict = {}
    for pert in perts:
        weights = pert_normalized_abs_scores_vsrest[pert]
        weights_total = weights.sum()
        if weights_total > 0:
            weights /= weights_total

        perturbation = adata.obs["perturbation"].astype(str)
        pert_mask = ~perturbation.apply(is_control).values

        mu_all = adata.X[pert_mask].mean(axis=0)
        mu_p, mu_p_hat = pseudobulk_means(adata, pert, pert_mask, preds_layer_name)
        delta = np.array(mu_p - mu_all).squeeze()
        delta_hat = np.array(mu_p_hat - mu_all).squeeze()

        pert_score = r2_score_on_deltas(delta, delta_hat, weights)
        r2_per_pert_dict[pert] = pert_score

    return np.mean(list(r2_per_pert_dict.values()))


# from diversity by design paper
def r2_score_on_deltas(delta_true, delta_pred, weights):
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan
    return r2_score(delta_true, delta_pred, sample_weight=weights)

if __name__ == "__main__":
    main()