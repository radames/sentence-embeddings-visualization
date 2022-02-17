import umap
import hdbscan
import copy

class UMAPReducer:
    def __init__(self, options={}):

        # set options with defaults
        options = {'n_components': 3, 'spread': 1, 'min_dist': 0.1, 'n_neighbors': 15,
                   'metric': 'hellinger', 'min_cluster_size': 60, 'min_samples': 15, **options}

        print(options)
        self.reducer = umap.UMAP(
            n_neighbors=options['n_neighbors'],
            min_dist=options['min_dist'],
            n_components=options['n_components'],
            metric=options['metric'],
            verbose=True)
        # cluster init
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=options['min_cluster_size'],
            min_samples=options['min_samples'],
            allow_single_cluster=True
        )
        self.cluster_params = copy.deepcopy(options)

    def setParams(self, options):
        # update params
        self.cluster_params = {**self.cluster_params, **options}

    def clusterAnalysis(self, data):
        clusters = self.clusterer.fit(data)
        return clusters

    def embed(self, data):
        result = self.reducer.fit_transform(data)
        return result
