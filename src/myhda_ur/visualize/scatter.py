import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


def plot_scatter(
    cluster: pd.Series = None,           
    embedding: tuple = None,
    title: str = None,
    show_label: bool = False
    ) -> None:
    """Plot scatter to visualize the clusters formed from the chosen clustering
       algorithm in a reduced dimension.
    
    Args:
        cluster:
            cluster result stored in HDAData.clusters
        embedding:
            embedding result stored in HDAData.embeddings
        title:
            title of the plot
        show_label: True / False
            display the gene / cell / traj label next to the data point
            
    """
    labels = cluster.index.tolist()
    cluster_labels = cluster.values
    
    coords, xlabel, ylabel = embedding
    
    if title is None:
        title = f'Clustering on {cluster.name}'
        
    cmap = plt.get_cmap("Paired")
    norm = mpl.colors.Normalize(vmin=0, vmax=9)          
    unique_labels = sorted(set(cluster_labels))
    color_map = {label: cmap(norm(label)) for label in unique_labels}
      
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for label in unique_labels:
        mask = cluster_labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=color_map[label],
            label=str(label)
        )

    if show_label:
        for i, name in enumerate(labels):
            ax.annotate(name, coords[i], fontsize=7, ha="left")
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ax.legend(
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True
    )
    
    plt.tight_layout()
    plt.show()