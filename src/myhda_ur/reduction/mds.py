from sklearn.manifold import MDS
import numpy as np
import pandas as pd

def reduce_mds(
    sim: pd.DataFrame
    ) -> tuple[np.ndarray, str, str]:
    """Dimensionality reduction of vectors to 2D using PCA.
    
    """
    dist = 1 - sim.values
    np.fill_diagonal(dist, 0)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    coords = mds.fit_transform(dist)
    xlabel, ylabel = "MDS1", "MDS2"
    
    return coords, xlabel, ylabel