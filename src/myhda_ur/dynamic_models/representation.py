from ..structure.object import HDAData
from ..dynamic_models.distance import _normalize_c
import numpy as np
import scipy.sparse as sp


def _get_gene_vectors(
    data: HDAData,
    key: str
    ) -> np.ndarray:
    """Build gene vectors based on key.
    
    Args:
        data: 
            (subset) data object to analyze
        key: "X" / "ntr" 
            gene expression or NTR value to create vectors from

    Returns:
        gene vectors
        
    """
    if key == "X":
        dense = data.X.toarray()
    
    elif key == "pca":
        raise ValueError('key="pca" is not valid for kind="gene". Only valid for kind="cell".')
    
    elif key in data.layers:
        dense = data.layers[key].toarray()

    else:
        raise ValueError(f"{key} not found in HDAData.")
    
    return dense


def _get_cell_vectors(
    data: HDAData,
    key: str,
    normalize: bool = True
    ) -> np.ndarray:
    """Build cell vectors based on key.
    
    Args:
        data:
            (subset) data object to analyze
        key: "X" / "ntr" 
            gene expression or NTR value to create vectors from
        normalize: True / False
            True = normalize if gene expression values used
            False = no normalization if ntr used
            
    Returns:
        cell vectors
        
    """
    if key == "X":
        dense = data.X.toarray()

    elif key in data.layers:
        dense = data.layers[key].toarray()

    else:
        raise ValueError(f"{key} not found in HDAData.")
    
    vectors = dense.T

    if normalize:
        vectors = _normalize_c(vectors)

    return vectors


def _build_trajectory_matrix(
    data: HDAData,
    key: sp.csr_matrix,
    trajectory_id: list
    ) -> tuple[np.ndarray, list[str]]: 
    """For each trajectory, extract gene expression or NTR values for all cells belonging to 
    that trajectory and flattens them into a single vector of shape (genes x timepoints).
    All trajectory vectors are stacked row-wise.

    Args:
        data: 
            (subset) data object to analyze
        key: data.X / data.layers["ntr"]
            gene expression or NTR matrix
        trajectory_id:
            list of trajectory IDs to create vectors from
    
    Returns:
        matrix of shape (trajectories x genes*timepoints), and labels
        
    """
    trajectory_id = list(trajectory_id)
    
    vectors = []
    labels = []
    
    # build one vector per trajectory
    for id in trajectory_id:
        traj = data.trajectories.loc[id]
        cells = traj.tolist()
        cell_idx = [data._cell_to_index[c] for c in cells]

        # extract gene x timepoints 
        sub = key[:, cell_idx]
        sub = sub.toarray()
        vec = sub.flatten()
        vectors.append(vec)
        labels.append(f'traj_{id}')
    
    # stack trajectories row-wise
    vectors = np.vstack(vectors)
    
    return vectors, labels


def _build_trajectory_matrix_pca(
    data: HDAData,
    pc_matrix: np.ndarray,
    trajectory_id: list
    ) -> tuple[np.ndarray, list[str]]:
    """Build trajectory vectors from a pre-computed cell x PC matrix.
    
    Args:
        data: 
            (subset) data object to analyze
        pc_matrix:
            ...
        trajectory_id:
            list of trajectory IDs to create vectors from
    
    Returns:
        matrix of shape: (n_trajectories, PCs*timepoints), and labels
        
    """
    trajectory_id = list(trajectory_id)
    
    vectors = []
    labels = []

    # build one vector per trajectory
    for id in trajectory_id:
        traj = data.trajectories.loc[id]
        cells = traj.tolist()
        cell_idx = [data._cell_to_index[c] for c in cells]

        # extract PC x timepoints 
        sub = pc_matrix[cell_idx, :].T    # transpose to (PCs x timepoints)
        vec = sub.flatten()               
        vectors.append(vec)
        labels.append(f'traj_{id}')

    return np.vstack(vectors), labels


def _get_trajectory_vectors(
    data: HDAData,
    key: str,
    trajectory_id: list,
    normalize: bool
    ):
    """Build trajectory vectors based on key.
    
    Args:
        data: 
            (subset) data object to analyze
        key: "X" / "ntr" 
            gene expression or NTR value to create vectors from
        trajectory_id:
            list of trajectory IDs to create vectors from
        normalize: True / False
            True = normalize if gene expression values used
            False = no normalization if ntr used
            
    Returns:
        trajectory vectors and labels
        
    """
    if key == "X":
        key = data.X
    
    elif key in data.layers:
        key = data.layers[key]
        
    else:
        raise ValueError(f"{key} not found in HDAData.")
    
    vectors, labels = _build_trajectory_matrix(data, key, trajectory_id)
    
    if normalize:
        vectors = _normalize_c(vectors)         
    
    return vectors, labels


def _get_vectors(
    data = None, 
    target = None,
    key = None, 
    kind = None, 
    trajectory_id = None
    ):
    """Helper to call the corresponding functions to create gene, cell or trajectory vectors.
    
    Args:
        data: 
            (subset) data object to analyze
        key: "X" / "ntr" / "pca_cell" / "pca_trajectory" / "new_rna" 
            gene expression or NTR value to create vectors from
        kind: "gene" / "cell" / "trajectory"
            type of vectors to create
        trajectory_id:
            list of trajectory IDs to create vectors from
    
    Returns:
        vectors and labels
        
    """
    if kind == "gene":
        vectors = _get_gene_vectors(data, key)
        labels = data.var.index.tolist()

    elif kind == "cell":
        if key == "pca_cell":
            vectors = target.layers["pca_cell"]  
            labels = data.obs.index.tolist()
            
        else:
            normalize = (key != "ntr")
            vectors = _get_cell_vectors(data, key, normalize)
            labels = data.obs.index.tolist()

    elif kind == "trajectory":
        if key == "pca_trajectory":
            vectors = target.layers["pca_trajectory"]
            labels = [f"traj_{id}" for id in trajectory_id]
            
        elif key == "pca_cell":
            vectors, labels = _build_trajectory_matrix_pca(
                                    data=data,
                                    pc_matrix=target.layers["pca_cell"],
                                    trajectory_id=trajectory_id
                                )
            
        else:
            normalize = (key == "X")
            vectors, labels = _get_trajectory_vectors(data, key, trajectory_id, normalize)

    return vectors, labels