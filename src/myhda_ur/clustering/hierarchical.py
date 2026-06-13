import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from ..structure.object import HDAData


def _get_linkage(
    sim: pd.DataFrame
    ) -> tuple:
    """Helper function to convert similarity matrix into distance matrix,
       condense it and compute linkage matrix using Ward's method.
    
        Args:
            sim: 
                similarity dataframe stored
                    
        Returns:
            Tuple of:
                - Z: Linkage matrix 
                - labels: List of sample labels from the similarity matrix index
            
    """
    labels = list(sim.index)
    dist = 1 - sim.values
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    
    return Z, labels


def cluster_hierarchical(  
    target: HDAData,    
    sim: pd.DataFrame,  
    kind: str = None,
    k: int = None,
    t: float = None
    ) -> HDAData:
    """Agglomerative hierarchical clustering on similarity matrix.

       Agglomerative clustering accepts a precomputed distance matrix directly,
       so the stored similarity matrix is first converted back to a distance matrix
       then into a linkage matrix. The clusters are formed using either a distance
       threshold t or fixed k clusters.
       
       Args:
            target: 
                location to store result - only for internal use by wrapper
            sim: 
                similarity dataframe stored
            kind: "gene" / "cell" / "trajectory"
                type of cluster elements
            k:
                number of clusters
            t: 
                distance threshold for cutting the dendogram
    
        Returns:
            object with cluster labels stored 
    
    """
    Z, labels = _get_linkage(sim)

    # extract clusters at a specific distance threshold
    if t is not None:
        clusters = fcluster(Z, t=t, criterion='distance')
        name = f"hierarchical_t{t}_{kind}" 
    
    # extract clusters with fixed k
    elif k is not None:
        clusters = fcluster(Z, t=k, criterion='maxclust')
        name = f"hierarchical_k{k}_{kind}" 
   
    else:
        raise ValueError("Provide either k clusters or t threshold distance.")
    
    cl = pd.Series(
            clusters - 1,   # back to 0-index
            index=labels,
            name=name
            )

    # store result
    target.clusters[name] = cl
    print(f'Cluster "{name}" stored! \n')

    return target