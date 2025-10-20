import sys
sys.dont_write_bytecode = True

import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score

def get_latent_clusterings(adata, latent_obsm_key, kmeans_K, n_neighbors=15, leiden_res=1, seed=2025):
    # Get a temporary adata to store clustering results for the given obsm key
    temp_adata = sc.AnnData(X=adata.obsm[latent_obsm_key])
    temp_adata.obs = adata.obs.copy()
    sc.pp.neighbors(temp_adata, use_rep="X", n_neighbors=n_neighbors, random_state=seed)

    # Leiden
    sc.tl.leiden(temp_adata, resolution=leiden_res)
    adata.obs[latent_obsm_key + "_leiden"] = temp_adata.obs["leiden"].values.astype("category")

    # Louain
    sc.tl.louvain(temp_adata)
    adata.obs[latent_obsm_key + "_louvain"] = temp_adata.obs["louvain"].values.astype("category")

    # K-means
    kmeans_model = KMeans(n_clusters=kmeans_K, random_state=seed)
    adata.obs[latent_obsm_key + "_kmeans"] = kmeans_model.fit_predict(adata.obsm[latent_obsm_key])
    adata.obs[latent_obsm_key + "_kmeans"] = adata.obs[latent_obsm_key + "_kmeans"].astype("category")

    return adata

def eval_knn_cv(adata, X_obsm_key, y_obs_key, knn_n_neighbors=5, num_cv=10):
    knn_model = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
    cv_scores = cross_val_score(
        estimator=knn_model,
        X=adata.obsm[X_obsm_key],
        y=adata.obs[y_obs_key],
        cv=num_cv
    )

    res_dict = {"KNN-CV": cv_scores.mean()}
    res_dict = {X_obsm_key: res_dict}

    return pd.DataFrame(
        res_dict
    ).T

def eval_clusterings(adata, cluster_obs_key, truth_obs_key):
    res_dict = dict()
    truth_labels = adata.obs[truth_obs_key]
    for cluster_key in cluster_obs_key:
        pred_labels = adata.obs[cluster_key]
        
        res_dict[cluster_key] = {
            "MI": mutual_info_score(truth_labels, pred_labels),
            "NMI": normalized_mutual_info_score(truth_labels, pred_labels),
            "AMI": adjusted_mutual_info_score(truth_labels, pred_labels),
            "ARI": adjusted_rand_score(truth_labels, pred_labels)
        }

    # Get the dataframe
    df = pd.DataFrame(res_dict).T
    return df

