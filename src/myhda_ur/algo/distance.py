from structure.object import HDAData
from scipy.spatial.distance import correlation, cosine
import numpy as np
import pandas as pd


def calc_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation distance between 2 vectors.
    
        Args:
            x: vector 1
            y: vector 2
        
        Returns:
            similarity in range [0, 1]
                - 1 = most similar  (perfect positive correlation, r = +1)
                - 0 = most dissimilar 
                
    """
    # Avoid NaN for zero variance
    if np.std(x) == 0 or np.std(y) == 0:   
        return 0.0
    dist = correlation(x, y) 
    return float(np.clip(1 - dist, 0, 1))


def calc_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance between 2 vectors.
    
        Args:
            x: vector 1
            y: vector 2
        
        Returns:
            similarity in range [0, 1]
                - 1 = most similar (identical direction, angle = 0°)
                - 0 = most dissimilar (angle = 90°)
                
    """
    # Avoid NaN for zero vector
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:   
        return 0.0
    dist = cosine(x, y) 
    return float(np.clip(1 - dist, 0, 1))


def normalize_c(vectors: np.ndarray, scale: float = 1e4) -> np.ndarray:
    """Normalize cell vectors by library size and log transform.
        Equivalent to: log( x_ij / library_size_j * scale + 1 )
        
        Args:
            vectors: (transposed) dense matrix (n_cells, n_genes)
            scale: scaling factor 
            
        Returns:
            normalized matrix
                
    """
    # Total counts per cell
    lib = vectors.sum(axis=1, keepdims=True) 
    # Avoid NaN for zero vector
    lib = np.where(lib == 0, 1, lib)
    return np.log1p(vectors / lib * scale)

    
metrics = {
    "correlation": calc_correlation,
    "cosine":      calc_cosine,
}


def compute_similarity(
    data: HDAData,
    key: str = None,
    axis:str = None,
    method: str = None,
    target: HDAData = None,
    name: str = None
    ) -> HDAData:
    """Compute similarity matrix.
    
        Args:
            data: subset of original matrix such as X or ntr to be analyzed
            key: the data from the subset to analyze (e.g "X" or "ntr")
            axis: gene or cell similarity ("gene" or "cell")
            method: metric method ("correlation" or "cosine")
       
        Returns:
            The same data object with new similarity matrix
            
    """
    if method not in metrics:
        raise ValueError(f'Unknwon method {method}. Choose one from {list(metrics.keys())}.')
    
    if axis not in ["gene", "cell"]:
        raise ValueError(f'Axis must be either "gene" or "cell.')
    
    metric = metrics[method]
    
    if key == "X":
        dense = data.X.toarray()
    elif key in data.layers:
        dense = data.layers[key].toarray()
    else:
        raise ValueError(f'{key} not found in HDAData object.')
    
    ## Case 1: Gene-gene correlation
    if axis == "gene":
        vectors = dense
        
    ## Case 2: Cell-cell correlation
    else:
        vectors = dense.T

        # normalize if gene expression 
        if key == "X":
            vectors = normalize_c(vectors)
    
    # Compute similarity matrix    
    n = vectors.shape[0]
    similarity = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity[i, j] = 1.0
            elif j < i:
                similarity[i, j] = similarity[j, i]         
            else:
                similarity[i, j] = metric(vectors[i], vectors[j])
 
    # Attach gene or cell names as labels for plotting convenience
    if axis == "gene":
        labels = data.var.index.tolist()
    else:
        labels = data.obs.index.tolist()
    
    similarity_df = pd.DataFrame(similarity, index=labels, columns=labels)
      
    # Feed result back into target data structure
    store = target if target is not None else data
    store.similarities[name] = similarity_df
    
    return store