from ..structure.object import HDAData
from ..dynamic_models.representation import _get_vectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(
    data: HDAData,
    cluster: pd.Series,           
    key: str,
    kind: str,
    trajectory_id: list = None,
    reduction: str = None,
    kmeans: bool = False
    ) -> None:
    
    labels = cluster.index.tolist()
    cluster_labels = cluster.values
    
    if reduction == "pca":
        if kmeans:
            vectors, _ = _get_vectors(
                                data=data,
                                key=key,
                                kind=kind,
                                trajectory_id=trajectory_id
                            )
            title = f'KMeans through PCA projection'
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors)
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)"
        
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
    