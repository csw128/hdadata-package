from ..structure.object import HDAData
from sklearn.manifold import MDS
import numpy as np
import pandas as pd


def reduce_MDS(
    target: HDAData,
    sim: pd.DataFrame,
    kind: str = None
    ) -> tuple[np.ndarray, str, str]:
    """Dimensionality reduction using metric MDS on precomputed similarity matrix.
    
        Args:
            target: 
                location to store result - only for internal use by wrapper
            sim: 
                similarity dataframe stored
            kind: "gene" / "cell" / "trajectory"
                type of cluster elements 
        
        Return:
            tuple:
                - x, y coordinates in reduced 2D MDS
                - xlabel: MDS1
                - ylabel: MDS2
                
    """
    dist = 1 - sim.values
    np.fill_diagonal(dist, 0)
    
    mds = MDS(
            n_components=2, 
            dissimilarity="precomputed", 
            random_state=0
            )
    
    coords = mds.fit_transform(dist)
    xlabel, ylabel = "MDS1", "MDS2"
    
    em = (coords, xlabel, ylabel)
    
    name = f"mds_{kind}"

    # store result
    target.embeddings[name] = em
    print(f'MDS embedding "{name}" stored! \n')

    return target