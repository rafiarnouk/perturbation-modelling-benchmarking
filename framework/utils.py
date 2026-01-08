import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import re

# annotates adata object to include split (either train/val/test)
def assign_splits(adata, split, exclude_ctrl=False):
    # figure out how many perts to put in each split
    unique_perts = list(adata.obs["perturbation"].unique())
    perts = [pert for pert in unique_perts if not re.search(r"ctrl|control", pert, flags=re.IGNORECASE)]
    ctrls = [pert for pert in unique_perts if re.search(r"ctrl|control", pert, flags=re.IGNORECASE)]
    n_perts_train, n_perts_val, n_perts_test = split_by_percentages(len(perts), list(split.values()))
    assert(len(perts) == n_perts_train + n_perts_val + n_perts_test)

    random.seed(12345) # set seed so we get the same split for a dataset every time (so models are evaluated fairly)
    random.shuffle(perts)
    perts_train = perts[:n_perts_train]
    perts_val = perts[n_perts_train:n_perts_train + n_perts_val]
    perts_test = perts[n_perts_train + n_perts_val:n_perts_train + n_perts_val + n_perts_test]

    # conditionally add controls to train split
    if not exclude_ctrl:
        perts_train.extend(ctrls)

    print(f"Splitting {len(perts)} perts and {len(ctrls)} controls: {perts_train=}, {perts_val=}, {perts_test=}")

    # segment data
    adata.obs.loc[adata.obs["perturbation"].isin(perts_train), "split"] = "train"
    adata.obs.loc[adata.obs["perturbation"].isin(perts_val), "split"] = "val"
    adata.obs.loc[adata.obs["perturbation"].isin(perts_test), "split"] = "test"

# not super robust, will fix if need arises
def assign_splits_proportional(adata, split):
    unique_perts_counts = adata.obs["perturbation"].value_counts().to_dict()
    pert_counts = {pert: count for pert, count in unique_perts_counts.items() if not re.search(r"ctrl|control", pert, flags=re.IGNORECASE)}
    ctrl_counts = {pert: count for pert, count in unique_perts_counts.items() if re.search(r"ctrl|control", pert, flags=re.IGNORECASE)}

    # weighted random pick until amount of cells in test passes threshold
    total_cells_count = sum(pert_counts.values())
    test_target = split["test"] * total_cells_count
    test_perts_count = 0
    possible_test_perts = list(pert_counts.keys())
    test_pert_weights = list(pert_counts.values())
    perts_test = []
    while test_perts_count < test_target:
        chosen_pert = random.choices(possible_test_perts, test_pert_weights)[0]
        perts_test.append(chosen_pert)
        test_perts_count += pert_counts[chosen_pert]

        # remove chosen pert from selection pool
        index = possible_test_perts.index(chosen_pert)
        possible_test_perts.pop(index)
        test_pert_weights.pop(index)
        print(f"chose {chosen_pert} for testing, so now {test_perts_count=}")
    
    # assign test perts
    adata.obs.loc[adata.obs["perturbation"].isin(perts_test), "split"] = "test"
    
    # assign controls to train
    ctrls = list(ctrl_counts.keys())
    adata.obs.loc[adata.obs["perturbation"].isin(ctrls), "split"] = "train"

    # split remaining cells (not perts) to train/val
    train_target = split["train"] * total_cells_count
    val_target = split["val"] * total_cells_count
    train_target -= sum(ctrl_counts.values())
    prob_train = train_target / (train_target + val_target)

    # randomly assign remaining cells to train or test
    mask_rest = adata.obs["split"].isna()
    rand_vals = np.random.rand(mask_rest.sum())
    adata.obs.loc[mask_rest, "split"] = np.where(rand_vals < prob_train, "train", "val")

# returns dictionary mapping perturbations found in adata object to genes being perturbed
def get_perturbed_genes_map(adata, separator="_", remove_metadata_tag=False):
    perts = list(adata.obs["perturbation"].unique())
    pert_map = {}
    for pert in perts:
        if remove_metadata_tag:
            pert_split = [gene for gene in pert.split(separator)[:-1] if gene != 'only']
        else:
            pert_split = [gene for gene in pert.split(separator) if gene != 'only']
        pert_map[pert] = pert_split
    return pert_map

# AI generated
def split_by_percentages(N, percentages):
    """
    Split a total number N into parts based on given percentages.

    Args:
        N (int or float): total amount to split
        percentages (list or tuple of floats): must sum to 1

    Returns:
        list of numbers (same length as percentages) that sum to N
    """
    if not abs(sum(percentages) - 1.0) < 1e-8:
        raise ValueError("Percentages must sum to 1")

    # initial split
    parts = [N * p for p in percentages]

    # if N is integer, adjust rounding to make sure the sum stays exact
    if isinstance(N, int):
        parts = [int(round(x)) for x in parts]
        diff = N - sum(parts)
        # fix rounding error by adjusting the largest parts
        for i in range(abs(diff)):
            parts[i % len(parts)] += 1 if diff > 0 else -1

    return parts

# adapted from https://github.com/shiftbioscience/diversity_by_design/tree/24a20c51a040ea489d6dd77cff14263c51c34a33
def compute_degs(adata, mode='vsrest', pval_threshold=0.05):
    """
    Compute differentially expressed genes (DEGs) for each perturbation.
    
    Args:
        adata: AnnData object with processed data
        mode: 'vsrest' or 'vscontrol'
            - 'vsrest': Compare each perturbation vs all other perturbations (excluding control)
            - 'vscontrol': Compare each perturbation vs control only
        pval_threshold: P-value threshold for significance (default: 0.05)
    
    Returns:
        dict: rank_genes_groups results dictionary
        
    Adds to adata.uns:
        - deg_dict_{mode}: Dictionary with perturbation as key and dict with 'up'/'down' DEGs as values
        - rank_genes_groups_{mode}: Full rank_genes_groups results
    """
    if mode == 'vsrest':
        # Remove control cells for vsrest analysis
        adata_subset = adata[
            ~adata.obs["perturbation"].str.contains(r"ctrl|control", case=False, regex=True)
        ].copy()
        reference = 'rest'
    elif mode == 'vscontrol':
        # Use full dataset for vscontrol analysis
        adata_subset = adata.copy()
        reference = 'control'
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")
    
    # ADDED: further reduce subset to remove perturbations with only one sample (otherwise sc.tl.rank_genes_groups fails)
    adata_subset = adata_subset[
        adata_subset.obs["perturbation"].map(
            adata_subset.obs["perturbation"].value_counts()
        ) >= 2
    ].copy()
    
    # Compute DEGs
    sc.tl.rank_genes_groups(adata_subset, 'perturbation', method='t-test_overestim_var', reference=reference)
    
    # Extract results
    names_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["pvals_adj"])
    logfc_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["logfoldchanges"])
    
    # For each perturbation, get the significant DEGs up and down regulated
    deg_dict = {}
    for pert in tqdm(adata_subset.obs['perturbation'].unique(), desc=f"Computing DEGs {mode}"):
        # not supporting vscontrol, would need to change following condition if we want to use vscontrol
        if mode == 'vscontrol' and pert == 'control':
            continue  # Skip control when comparing vs control
            
        pert_degs = names_df[pert]
        pert_pvals = pvals_adj_df[pert]
        pert_logfc = logfc_df[pert]
        
        # Get significant DEGs
        significant_mask = pert_pvals < pval_threshold
        pert_degs_sig = pert_degs[significant_mask]
        pert_logfc_sig = pert_logfc[significant_mask]
        
        # Split into up and down regulated
        pert_degs_sig_up = pert_degs_sig[pert_logfc_sig > 0].tolist()
        pert_degs_sig_down = pert_degs_sig[pert_logfc_sig < 0].tolist()
        
        deg_dict[pert] = {'up': pert_degs_sig_up, 'down': pert_degs_sig_down}
    
    # Save results to adata.uns
    adata.uns[f'deg_dict_{mode}'] = deg_dict
    adata.uns[f'rank_genes_groups_{mode}'] = adata_subset.uns['rank_genes_groups'].copy()
    
    return adata_subset.uns['rank_genes_groups']
    