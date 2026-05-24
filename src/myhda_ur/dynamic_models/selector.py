from structure.object import HDAData
import numpy as np


def get_highly_variable_genes(
    input: HDAData,
    n: int
    ) -> list:
    """ 
    Identify highly variable genes by computing the NTR variance across cells 
    for each gene. Then identify genes with the highest variances.
    
    """
    dense = input.layers["ntr"].toarray()
    gene_var = np.var(dense, axis=1)
    top_idx = np.argsort(gene_var)[::-1][:n]
    selected_genes = [input._index_to_gene[idx] for idx in top_idx]
    
    return selected_genes


def get_timepoint_cells(
    input: HDAData,
    n_samples_per_timepoint: int
    ) -> list:
    """ 
    Get random number of samples of cells per timepoint from the cell barcodes table.
    """
    tp_cells = []
    timepoints = ["0h", "2h", "4h", "6h", "8h", "10h"]
    for tp in timepoints:
        filtered_df = input.obs.loc[input.obs["time"]==tp]
        sampled_df = filtered_df.sample(n=n_samples_per_timepoint, random_state=42)
        sampled_cells = sampled_df.index.tolist()
        tp_cells.extend(sampled_cells)
    
    return tp_cells


def get_highly_dynamic_trajectories(
    input: HDAData,
    n: int
    ) -> list:
    
    traj_index = input.trajectories.index.tolist()
    key = input.layers["ntr"]
    
    vectors = []
    
    for id in range(1, len(traj_index)+1):
        traj = input.trajectories.loc[id]
        cells = traj.tolist()
        cell_idx = [input._cell_to_index[c] for c in cells]
        
        sub = key[:, cell_idx]
        sub = sub.toarray()
        vec = sub.flatten()
        vectors.append(vec)
    
    vectors = np.array(vectors)
    
    # get variance in each traj vector
    var = np.var(vectors, axis=1)
    top_idx = np.argsort(var)[::-1][:n]
    selected_traj = [traj_index[idx] for idx in top_idx]
    
    return selected_traj