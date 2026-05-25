from ..structure.object import HDAData
import numpy as np


def get_highly_variable_genes(
    data: HDAData,
    n: int
    ) -> list:
    """ 
    Identify highly variable genes (HVG) by computing the NTR variance across cells for each gene. 
    NTR variance are sorted decresingly to return the top n HVGs.
    
    Args:
        data: 
            data object to retrieve ntr values from
        n: 
            number of top n HVG
    
    Returns:
        list of top n HVG
    
    """
    dense = data.layers["ntr"].toarray()
    gene_var = np.var(dense, axis=1)
    top_idx = np.argsort(gene_var)[::-1][:n]
    selected_genes = [data._index_to_gene[idx] for idx in top_idx]
    
    return selected_genes


def get_timepoint_cells(
    data: HDAData,
    n: int
    ) -> list:
    """ 
    Get random number of cell samples per timepoint from the cell barcodes table (obs).
    
    Args:
        data: 
            data object to retrieve cell names from
        n: 
            number of samples to extract per timepoint
    
    Returns:
        list of timepoint cell samples
    
    """
    tp_cells = []
    timepoints = ["0h", "2h", "4h", "6h", "8h", "10h"]
    for tp in timepoints:
        filtered_df = data.obs.loc[data.obs["time"]==tp]
        sampled_df = filtered_df.sample(n=n, random_state=42)
        sampled_cells = sampled_df.index.tolist()
        tp_cells.extend(sampled_cells)
    
    return tp_cells


def get_highly_dynamic_trajectories(
    data: HDAData,
    n: int
    ) -> list:
    """ 
    Identify highly dynamic trajectories by computing the NTR variance across timepoint for HVGs. 
    NTR variance are sorted decresingly to return the top n trajectories.
    
    Args:
        data: 
            data object to retrieve trajectories from
        n: 
            number of top n trajectories
    
    Returns:
        list of top n trajectories index
    
    """
    traj_index = data.trajectories.index.tolist()
    key = data.layers["ntr"]
    
    vectors = []
    
    for id in range(1, len(traj_index)+1):
        traj = data.trajectories.loc[id]
        cells = traj.tolist()
        cell_idx = [data._cell_to_index[c] for c in cells]
        
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