from scipy.spatial.distance import correlation, cosine
import numpy as np


def _calc_correlation(
    x: np.ndarray, 
    y: np.ndarray
    ) -> float:
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


def _calc_cosine(
    x: np.ndarray, 
    y: np.ndarray
    ) -> float:
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


def _normalize_c(
    vectors: np.ndarray, 
    scale: float = 1e4
    ) -> np.ndarray:
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
