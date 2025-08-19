# pose_matcher.py
import numpy as np
from sklearn.neighbors import NearestNeighbors

class PoseMatcher:
    """
    KD-tree/ball-tree wrapper for fast nearest-neighbour search over poses.

    Expected by run_pipeline.py:
      - constructor: PoseMatcher(pose_matrix, n_neighbors=1, metric='euclidean')
      - query: returns (distances, indices) for a 1xD query pose vector
    """
    def __init__(self, pose_matrix, n_neighbors=1, metric='euclidean'):
        """
        pose_matrix : (N, D) numpy array
        n_neighbors : int
        metric      : str ('euclidean' by default)
        """
        self.pose_matrix = np.asarray(pose_matrix, dtype=np.float32)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric

        # build NN model
        self.nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='auto',
            metric=self.metric
        )
        self.nn.fit(self.pose_matrix)

    def query(self, pose_vector, k=None):
        """
        pose_vector : (D,) array-like
        k           : optional override for number of neighbors

        Returns:
            distances : (k,) float array
            indices   : (k,) int array
        """
        if k is None:
            k = self.n_neighbors
        q = np.asarray(pose_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.nn.kneighbors(q, n_neighbors=k, return_distance=True)
        return distances[0], indices[0]
