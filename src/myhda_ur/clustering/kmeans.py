from ..structure.object import HDAData
from ..dynamic_models.representation import _get_vectors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')  


def find_kmeans_k(
    data: HDAData,
    key: str,
    kind: str,
    trajectory_id: list = None,
    max_clusters: int = 6
    ) -> None:
    """Plot inertia vs k to find the optimal k for KMeans using the Elbow Method.
        
        Args:
            data: 
                (subset) data object to analyze
            key: "X" / "ntr" 
                values to compute on 
            kind: "gene" / "cell" / "trajectory"
                type of cluster elements
            trajectory_id: 
                selected trajectories to analyze
            max_clusters:
                maximum number of clusters to form
    
        Returns:
           Line plot of intertia vs k
                
    """
    vectors, _ = _get_vectors(
                            data=data,
                            key=key,
                            kind=kind,
                            trajectory_id=trajectory_id
                        )

    inertias = []

    for k in range(2, max_clusters):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init="auto")
        kmeans.fit(vectors)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(2, max_clusters), inertias, marker='o')
    plt.title('Inertia vs k clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def cluster_kmeans(
    data: HDAData,
    target: HDAData,
    key: str = None,
    kind: str = None,
    trajectory_id: list = None,
    k: int = 3
    ) -> HDAData:
    """KMeans clustering on gene/cell/trajectory vectors.
    
       sklearn KMeans does not work directly on pre-computed similarity/distance matrix.
       It requires the raw vectors to compute its own internal euclidean distance for clustering.
        
        Args:
            data: 
                (subset) data object to analyze
            target: 
                location to store result - only for internal use by wrapper
            key: "X" / "ntr" 
                values to compute on 
            kind: "gene" / "cell" / "trajectory"
                type of cluster elements
            trajectory_id: 
                selected trajectories to analyze
            k:
                number of clusters
    
        Returns:
           object with cluster labels stored 
                
    """
    vectors, labels = _get_vectors(
                        data=data,
                        key=key,
                        kind=kind,
                        trajectory_id=trajectory_id
                    )
    
    kmeans = KMeans(
                n_clusters=k, 
                init='k-means++', 
                n_init='auto', 
                random_state=0
                )
    kmeans.fit(vectors)
    
    name = f"kmeans_k{k}_{kind}"
    
    cl = pd.Series(
            kmeans.labels_,
            index=labels,
            name=name
            )

    # store result
    target.clusters[name] = cl
    print(f'Cluster "{name}" stored! \n')
    
    return target
