from structure.object import HDAData
from dynamic_models.distance import _normalize_c
import numpy as np


def _build_mean_ntr_matrix(
    input: HDAData
    ):
    
    ntr_hvg = input.layers["ntr"].toarray()
    
    trajectories = input.trajectories
    timepoints = trajectories.columns.tolist()
    
    mean_ntr_matrix = np.zeros((ntr_hvg.shape[0], len(timepoints)))
    
    for col_idx, tp in enumerate(timepoints):
        cells = trajectories[tp].tolist()
        cell_indices = [input._cell_to_index[c] for c in cells]
        # mean NTR across all cells in the trajectory time point for each HVG
        mean_ntr_matrix[:, col_idx] = ntr_hvg[:, cell_indices].mean(axis=1)

    return mean_ntr_matrix 
    

def _get_gene_vectors(
    input: HDAData,
    key: str
    ) -> np.ndarray:
    
    if key == "X":
        dense = input.X.toarray()
        
    elif key == "mean_ntr":
        dense = _build_mean_ntr_matrix(input)

    elif key in input.layers:
        dense = input.layers[key].toarray()

    else:
        raise ValueError(f"{key} not found in HDAData.")
    
    return dense


def _get_cell_vectors(
    input: HDAData,
    key: str,
    normalize: bool = True
    ) -> np.ndarray:
    
    if key == "X":
        dense = input.X.toarray()

    elif key in input.layers:
        dense = input.layers[key].toarray()

    else:
        raise ValueError(f"{key} not found in HDAData.")
    
    vectors = dense.T

    if normalize:
        vectors = _normalize_c(vectors)

    return vectors


def _build_trajectory_matrix(
    input: HDAData,
    key,
    trajectory_id
    ):
    
    trajectory_id = list(trajectory_id)
    
    vectors = []
    labels = []
    
    # build one vector per trajectory
    for id in trajectory_id:
        traj = input.trajectories.loc[id]
        cells = traj.tolist()
        cell_idx = [input._cell_to_index[c] for c in cells]

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
    input: HDAData,
    key: str,
    trajectory_id,
    normalize
    ):
    
    if key == "X":
        key = input.X
    
    elif key in input.layers:
        key = input.layers[key]
        
    else:
        raise ValueError(f"{key} not found in HDAData.")
    
    vectors, labels = _build_trajectory_matrix(input, key, trajectory_id)
    
    if normalize:
        vectors = _normalize_c(vectors)         
    
    return vectors, labels


def _get_vectors(
    input: HDAData,
    key: str,
    kind: str,
    trajectory_id,
    ):
    
    if kind == "gene":

        vectors = _get_gene_vectors(input, key)
        labels = input.var.index.tolist()

    elif kind == "cell":

        normalize = (key == "X")
        vectors = _get_cell_vectors(input, key, normalize)
        labels = input.obs.index.tolist()
        
    elif kind == "trajectory":
        normalize = (key == "X")
        vectors, labels = _get_trajectory_vectors(input, key, trajectory_id, normalize)
        
    else:
        raise ValueError(f'Unknown kind "{kind}". Must be either "gene", "cell", or "trajectory".')

    return vectors, labels
