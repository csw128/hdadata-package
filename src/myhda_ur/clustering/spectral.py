import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import squareform
from ..structure.object import HDAData


def cluster_spectral(  
    target: HDAData,    
    sim: pd.DataFrame,  
    kind: str = None,
    k: int = None
    ) -> HDAData:
    """Spectral (graph-based) clustering
    
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