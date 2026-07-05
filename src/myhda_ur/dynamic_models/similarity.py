from ..structure.object import HDAData
from ..dynamic_models.distance import _calc_correlation, _calc_cosine
from ..dynamic_models.representation import _get_vectors
import numpy as np
import pandas as pd
from collections.abc import Callable


metrics = {
    "correlation": _calc_correlation,
    "cosine":      _calc_cosine,
    }


def _pairwise_similarity(
    vectors: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float]
    ) -> np.ndarray:
    """Compute pairwise similarity matrix based on chosen distance metric.
    
    Args:
        vectors: 
            vectors to compare distances
        metric: _calc_correlation / _calc_cosine
            distance function to be called 
    
    Returns:
        pairwise similarity matrix with the same pairs having highest similarity of 1.0 on the diagonal.
        
    """
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

    return similarity


def compute_similarity(
    data: HDAData = None,
    target: HDAData = None,
    key: str = None,
    kind: str = None,
    trajectory_id: list = None,
    method: str = None,
    ) -> HDAData:
    """Helper function to build vectors and labels into similarity matrix, and feeds the result 
       back into the original data structure.
    
    Args:
        data: 
            (subset) data object to analyze
        target: 
            location to store result - only for internal use by wrapper
        key: "X" / "ntr" / "pca_cell" / "pca_trajectory"
            data from the subset to analyze 
        kind: "gene" / "cell" / "trajectory"
            type of similarity 
        trajectory_id: 
            selected trajectories to analyze
        method: "correlation" / "cosine"
            distance metric
    
    Returns:
        object with similarity matrix stored 
            
    """
    # validate metric
    if method not in metrics:
        raise ValueError(f'Unknwon method {method}. Choose one from {list(metrics.keys())}.')

    metric = metrics[method]

    vectors, labels = _get_vectors(
                            data=data,
                            target=target,
                            key=key,
                            kind=kind,
                            trajectory_id=trajectory_id
                        )

    name = f"{method}_{key}_{kind}"

    similarity = _pairwise_similarity(vectors=vectors, metric=metric)
    similarity_df = pd.DataFrame(similarity, index=labels, columns=labels)

    target.similarities[name] = similarity_df
    print(f'Similarity "{name}" stored!')
    return target