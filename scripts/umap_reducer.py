# from https://huggingface.co/spaces/radames/sentence-embeddings-visualization/tree/main
import umap
import hdbscan
import copy


class UMAPReducer:
    def __init__(self, umap_options={}, cluster_options={}):

        # set options with defaults
        self.umap_options = {'n_components': 3, 'spread': 1, 'min_dist': 0.1, 'n_neighbors': 15,
                             'metric': 'cosine', "verbose": True, **umap_options}
        self.cluster_options = {'allow_single_cluster': True, 'min_cluster_size': 500, 'min_samples': 10, **cluster_options}

    def setParams(self, umap_options={}, cluster_options={}):
        # update params
        self.umap_options = {**self.umap_options, **umap_options}
        self.cluster_options = {**self.cluster_options, **cluster_options}

    def clusterAnalysis(self, data):
        print("Cluster params:", self.cluster_options)
        clusters = hdbscan.HDBSCAN().fit(data)  # **self.cluster_options
        return clusters

    def embed(self, data):
        print("UMAP params:", self.umap_options)
        result = umap.UMAP(**self.umap_options).fit_transform(data)
        return result
