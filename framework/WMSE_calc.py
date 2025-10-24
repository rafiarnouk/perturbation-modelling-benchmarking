import numpy as np
import pandas as pd

def weighted_delta_wmse(means_true, means_pred, control, gene_weights=None):
   #checking alignment
    genes = means_true.columns.intersection(means_pred.columns)
    conds = means_true.index.intersection(means_pred.index)
    if len(genes) == 0 or len(conds) == 0:
        raise ValueError("No overlapping genes or conditions between truth and preds.")
    mt = means_true.loc[conds, genes]
    mp = means_pred.loc[conds, genes]

   
    d_true = mt.subtract(mt.loc[control], axis=1)
    d_pred = mp.subtract(mp.loc[control], axis=1)

    if gene_weights is None:
        w = np.full(len(genes), 1.0 / len(genes), dtype=float)
    elif isinstance(gene_weights, pd.Series):
        w = gene_weights.reindex(genes).fillna(0).to_numpy(dtype=float)
    else:
        w = np.asarray(gene_weights, dtype=float)
        if w.shape[0] != len(genes):
            raise ValueError("gene_weights length must match number of genes after alignment.")
    w = np.nan_to_num(w, nan=0.0)
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / len(w)
    else:
        w /= s
    #checks gene per condition 
    rows = []
    for c in d_true.index:
        if c == control:
            continue
        err2 = (d_pred.loc[c].to_numpy() - d_true.loc[c].to_numpy())**2
        wmse = float(np.nansum(w * err2))
        rows.append({"condition": c, "wmse": wmse})

    return pd.DataFrame(rows)