import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from ..structure.object import HDAData


def cluster_spectral(  
    target: HDAData,    
    sim: pd.DataFrame,  
    kind: str = None,
    k: int = None
    ) -> HDAData:
    """Spectral (graph-based) clustering on similarity matrix.
    
       sklearn SpectralClustering accepts a precomputed affinity matrix. 
       The stored similarity matrix is treated as the affinity graph by clipping the values 
       to zero to ensure non-negative edge weights. Spectral clustering is then applied to 
       group elements based on their connectivity.
       
       Args:
            target: 
                location to store result - only for internal use by wrapper
            sim: 
                similarity dataframe stored
            kind: "gene" / "cell" / "trajectory"
                type of cluster elements
            k:
                number of clusters

        Returns:
            object with cluster labels stored 
            
    """
    labels = list(sim.index)
    affinity = np.clip(sim.values, 0, None)
    np.fill_diagonal(affinity, 0)

    clustering = SpectralClustering(
                    n_clusters=k,
                    affinity='precomputed',   
                    assign_labels='discretize',
                    random_state=0
                )
    clustering.fit(affinity)
    
    name = f"spectral_k{k}_{kind}"
    
    cl = pd.Series(
            clustering.labels_,   
            index=labels,
            name=name
            )

    # store result
    target.clusters[name] = cl
    print(f'Cluster "{name}" stored! \n')
 
    return target