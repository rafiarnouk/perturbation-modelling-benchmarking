import argparse
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np

# example usage: 
"""
python metric_calculator.py \
  --preds_files adamson_pred_sclambda_0 adamson_pred_sclambda_1 adamson_pred_sclambda_2 \
  --metrics pearson \
  --preds_layer_name pred_sclambda

for baselines:
python metric_calculator.py \
  --preds_files adamson_pred_baseline_0 adamson_pred_baseline_1 adamson_pred_baseline_2 \
  --metrics pearson \
  --preds_layer_name pred_baseline_mean
"""

def main():
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset")
    parser.add_argument("--preds_files", nargs="*", type=str, help="Names of the files (should be in data/predictions)")
    parser.add_argument("--metrics", nargs="*", type=str, default=["pearson"], help="Metrics to calculate")
    parser.add_argument("--preds_layer_name", type=str, default=["pred"], help="Name of the predictions layer in the adata object")
    args = parser.parse_args()
    files = args.preds_files
    metrics = args.metrics
    preds_layer_name = args.preds_layer_name

    metric_dict = {metric: [] for metric in metrics}

    for split_number, file in enumerate(files):
        adata = sc.read(f"../data/predictions/{file}.h5ad")
        test_rows = ~np.all(np.isnan(adata.layers[preds_layer_name]), axis=1)
        actual = adata.X.toarray()[test_rows]
        pred = adata.layers[preds_layer_name][test_rows]

        if "pearson" in metrics:
            corr = calculate_pearson(actual, pred)
            print(f"Mean pearson correlation across cells for split {split_number}:", corr)
            metric_dict["pearson"].append(corr)

    for metric, results in metric_dict.items():
        print(f"Average {metric} across all splits:", sum(results) / len(results))

def calculate_pearson(actual, pred):
    correlations = np.array([
        np.corrcoef(actual[i, :], pred[i, :])[0, 1]
        for i in range(actual.shape[0])
    ])
    return np.nanmean(correlations)

    # histogram code
    # plt.hist(correlations, bins=50)
    # plt.xlabel("Per-cell correlation")
    # plt.ylabel("Number of cells")
    # plt.title("Predicted vs Actual Expression Correlation (per cell)")
    # plt.show()

if __name__ == "__main__":
    main()