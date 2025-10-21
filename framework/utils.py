import scanpy as sc
import random

def get_random_split(adata, split):
    # figure out how many perts to put in each split
    perts = list(adata.obs["perturbation"].unique())
    n_perts_train, n_perts_val, n_perts_test = split_by_percentages(len(perts), list(split.values()))
    assert(len(perts) == n_perts_train + n_perts_val + n_perts_test)

    random.seed(12345) # set seed so we get the same split for a dataset every time (so models are evaluated fairly)
    random.shuffle(perts)
    perts_train = perts[:n_perts_train]
    perts_val = perts[n_perts_train:n_perts_train + n_perts_val]
    perts_test = perts[n_perts_train + n_perts_val:n_perts_train + n_perts_val + n_perts_test]

    print(f"Splitting {len(perts)} perts: {perts_train=}, {perts_val=}, {perts_test=}")

    # segment data and save
    adata_train = adata[adata.obs["perturbation"].isin(perts_train)].copy()
    adata_val = adata[adata.obs["perturbation"].isin(perts_val)].copy()
    adata_test = adata[adata.obs["perturbation"].isin(perts_test)].copy()

    return adata_train, adata_val, adata_test

# ChatGPT generated
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