from structure.object import HDAData
from dynamic_models.distance import _calc_correlation, _calc_cosine
from dynamic_models.representation import _get_vectors
import numpy as np
import pandas as pd


metrics = {
    "correlation": _calc_correlation,
    "cosine":      _calc_cosine,
    }


def _pairwise_similarity(
    vectors: np.ndarray,
    metric
    ) -> np.ndarray:
    
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
    data: HDAData,
    key: str,
    kind: str,
    trajectory_id = None,
    method: str = None,
    target: HDAData = None,
    name: str = None
    ) -> HDAData:
    """Compute similarity matrix.
    
        Args:
            data: subset of original matrix such as X or ntr to be analyzed
            key: the data from the subset to analyze (e.g "X" or "ntr")
            kind: gene / cell / trajectory similarity ("gene" or "cell" or "trajectory")
            method: metric method ("correlation" or "cosine")
            target: object to store similarity matrix
            name: name of similarity matrix to store
       
        Returns:
            target object with new similarity matrix
            
    """
    # validate metric
    if method not in metrics:
        raise ValueError(f'Unknwon method {method}. Choose one from {list(metrics.keys())}.')

    metric = metrics[method]

    # build vectors
    vectors, labels = _get_vectors(
                            data=data,
                            key=key,
                            kind=kind,
                            trajectory_id=trajectory_id
                        )

    # compute similarity matrix
    similarity = _pairwise_similarity(
                        vectors=vectors,
                        metric=metric
                    )

    similarity_df = pd.DataFrame(
                        similarity,
                        index=labels,
                        columns=labels
                    )

    # store result
    store = target if target is not None else data

    if name is None:
        name = f"{kind}_{method}_{key}"

    store.similarities[name] = similarity_df

    return store