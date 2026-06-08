from ..structure.object import HDAData
from ..dynamic_models.representation import _get_vectors
from ..reduction.pca import reduce_pca
from ..reduction.mds import reduce_mds
import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(
    data: HDAData = None,
    cluster: pd.Series = None,           
    key: str = None,
    kind: str = None,
    trajectory_id: list = None,
    sim: pd.DataFrame = None
    ) -> None:
    
    labels = cluster.index.tolist()
    cluster_labels = cluster.values
    
    # MDS for hierarchical & graph-based 
    if sim is not None:
        coords, xlabel, ylabel = reduce_mds(sim)
        title = f'{cluster.name} - MDS projection'
    
    # PCA for kmeans
    else:
        vectors, _ = _get_vectors(
                            data=data,
                            key=key,
                            kind=kind,
                            trajectory_id=trajectory_id
                        )
        
        coords, xlabel, ylabel = reduce_pca(vectors)
        title = f'{cluster.name} - PCA projection'
        
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=cluster_labels, cmap="tab10"
        )
    
    for i, name in enumerate(labels):
        ax.annotate(name, coords[i], fontsize=7, ha="left")
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()