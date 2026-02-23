pca_dict = {
    "pca_components": 120,
}

umap_dict = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "euclidean",
    "spread": 1.0,
    "low_memory": True,
    "init": "spectral",
}

DIM_ALGORITHMS = {
    "pca": pca_dict,
    "umap": umap_dict,
}