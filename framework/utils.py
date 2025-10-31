import scanpy as sc
import numpy as np
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
def assign_splits_proportional(adata, split, seed=0):
    random.seed(seed)
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
        print(f"chose {chosen_pert}, so now {test_perts_count=}")
    
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

    # mask for rows not in test
    mask_rest = adata.obs["split"].isna()

    # generate random numbers for remaining rows
    rand_vals = np.random.rand(mask_rest.sum())

    # assign train/val based on probability x
    adata.obs.loc[mask_rest, "split"] = np.where(rand_vals < prob_train, "train", "val")

# returns dictionary mapping perturbations found in adata object to genes being perturbed
def get_perturbed_genes_map(adata, separator="_"):
    perts = list(adata.obs["perturbation"].unique())
    pert_map = {}
    for pert in perts:
        pert_split = [gene for gene in pert.split(separator)[:-1] if gene != 'only']
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