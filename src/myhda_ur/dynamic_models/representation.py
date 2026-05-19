from structure.object import HDAData
from dynamic_models.distance import _normalize_c
import numpy as np


def _build_mean_ntr_matrix(
    data: HDAData
    ):
    
    ntr_hvg = data.layers["ntr"].toarray()
    
    trajectories = data.trajectories
    timepoints = trajectories.columns.tolist()
    
    mean_ntr_matrix = np.zeros((ntr_hvg.shape[0], len(timepoints)))
    
    for col_idx, tp in enumerate(timepoints):
        cells = trajectories[tp].tolist()
        cell_indices = [data._cell_to_index[c] for c in cells]
        # mean NTR across all cells in the trajectory time point for each HVG
        mean_ntr_matrix[:, col_idx] = ntr_hvg[:, cell_indices].mean(axis=1)

    return mean_ntr_matrix 
    

def _get_gene_vectors(
    data: HDAData,
    key: str
    ) -> np.ndarray:
    
    if key == "X":
        dense = data.X.toarray()
        
    if key == "mean_ntr":
        dense = _build_mean_ntr_matrix(data)

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
    key,
    trajectory_id
    ):
    
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
    

def _get_trajectory_vectors(
    data: HDAData,
    key: str,
    trajectory_id,
    normalize
    ):
    
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
    data: HDAData,
    key: str,
    kind: str,
    trajectory_id,
    ):
    
    if kind == "gene":

        vectors = _get_gene_vectors(data, key)
        labels = data.var.index.tolist()

    elif kind == "cell":

        normalize = (key == "X")
        vectors = _get_cell_vectors(data, key, normalize)
        labels = data.obs.index.tolist()
        
    elif kind == "trajectory":
        normalize = (key == "X")
        vectors, labels = _get_trajectory_vectors(data, key, trajectory_id, normalize)
        
    else:
        raise ValueError(f'Unknown kind "{kind}". Must be either "gene", "cell", or "trajectory".')

    return vectors, labels
