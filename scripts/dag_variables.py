BUCKET_NAME = "mri-dataset"
PROCESSED_PREFIX = "mri"
LOCAL_DATA_DIR = "mri_train_data"

pca_dict = {
    "pca_components": 120,
}

tsne_dict = {
    "n_components": 2,
    "perplexity": 30,
    "early_exaggeration": 12,
    "learning_rate": "auto",
    "metric": "euclidean",
    "init": "pca",
    "angle": 0.5,
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
    "tsne": tsne_dict,
    "umap": umap_dict,
}