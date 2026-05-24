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
            similarity in range [-1, 1]
                - 1 = most similar  
                - -1 = most dissimilar 
                
    """
    # avoid NaN for zero vector
    if np.std(x) == 0 or np.std(y) == 0:   
        return 0.0
    sim = 1 - correlation(x, y)   # correlation in [0, 2], sim in [-1, 1]
    return sim


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
                - 1 = most similar 
                - -1 = most dissimilar
                
    """
    # avoid NaN for zero vector
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:   
        return 0.0
    sim = 1 - cosine(x, y)      # cosine in [0, 2], sim in [-1, 1]
    return sim


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
    # total counts per cell
    lib = vectors.sum(axis=1, keepdims=True) 
    # avoid NaN for zero vector
    lib = np.where(lib == 0, 1, lib)
    return np.log1p(vectors / lib * scale)
