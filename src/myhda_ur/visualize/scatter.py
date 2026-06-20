from ..structure.object import HDAData
import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(
    cluster: pd.Series = None,           
    embedding: tuple = None,
    reduction: str = None,
    show_label: bool = False
    ) -> None:
    """Plot scatter to visualize the clusters formed from the chosen clustering
       algorithm in a reduced dimension.
    
    Args:
        cluster:
            cluster result stored in HDAData.clusters
        embedding:
            embedding result stored in HDAData.embeddings
        reduction: "PCA" / "MDS"
            type of dimension reduction algorithm used
        show_label: True / False
            display the gene / cell / traj label next to the data point
            
    """
    labels = cluster.index.tolist()
    cluster_labels = cluster.values
    
    coords, xlabel, ylabel = embedding
    
    title = f'{cluster.name} - {reduction} projection'
      
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=cluster_labels, cmap="tab10"
        )
    
    if show_label:
        for i, name in enumerate(labels):
            ax.annotate(name, coords[i], fontsize=7, ha="left")
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()