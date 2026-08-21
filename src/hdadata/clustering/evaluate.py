import numpy as np
import pandas as pd
from ..dynamic_models.representation import _get_vectors
from sklearn.metrics import silhouette_score


def calc_cl_silhouette_score(
    cl: pd.Series,
    precomputed: bool,
    sim: pd.DataFrame = None,
    data = None,
    key: str = None,
    kind: str = None,
    trajectory_id: list = None
    ) -> float:
    """Compute Silhouette Score for a clustering result (independent of clustering algorithm).
        Silhouette Score is computed from the distance between a point, its cluster, and the nearest cluster.
        Score ranges (+1 to -1) as best clusters to worst. Values near 0 indicate overlapping clusters. 

       Args:
            cl:
                cluster labels
            precomputed:
                True:  if sim matrix is used (spectral/hierarchical)
                False: if raw vectors required (kmeans)
            sim:
                similarity matrix
            
            The following arguments are only needed for kmeans, where precomputed=False:
            
            data: 
                (subset) data object to analyze
            key: "X" / "ntr" / "new_rna" / "pca_cell" / "pca_trajectory" 
                values to compute on 
            kind: "gene" / "cell" / "trajectory"
                type of cluster elements
            trajectory_id: 
                selected trajectories to analyze
        
        Returns:
            float Silhouette Score
        
    """
    # Case 1: spectral / hierarchical
    if precomputed:
        if sim is None:
            raise ValueError("`sim` is required when precomputed=True for Spectral / Hierarchical Clustering")

        common = cl.index.intersection(sim.index)
        sim_ = sim.loc[common, common]
        cl_ = cl.loc[common]

        dist = 1 - np.clip(sim_.values, 0, 1)
        np.fill_diagonal(dist, 0)

        if len(set(cl_)) < 2:
            print("Warning: fewer than 2 clusters present, cannot compute score.")
            return np.nan

        score = silhouette_score(dist, cl_.values, metric="precomputed")

    # Case 2: kmeans
    else:
        if (data is None) or (key is None) or (kind is None) or (trajectory_id is None):
            raise ValueError(" 'data', 'key', 'kind' are required when precomputed=False for KMeans Clustering")
                
        vectors, labels = _get_vectors(
            data=data,
            key=key,
            kind=kind,
            trajectory_id=trajectory_id
        )
        vectors_ = pd.DataFrame(vectors, index=labels)

        common = cl.index.intersection(vectors_.index)
        vectors_ = vectors_.loc[common]
        cl_ = cl.loc[common]

        if len(set(cl_)) < 2:
            print("Warning: fewer than 2 clusters present, cannot compute score.")
            return np.nan

        score = silhouette_score(vectors_.values, cl_.values, metric="euclidean")
    
    return score