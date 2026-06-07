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
    ):

    vectors, labels = _get_vectors(
                        data=data,
                        key=key,
                        kind=kind,
                        trajectory_id=trajectory_id
                    )
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=0)
    kmeans.fit(vectors)
    
    name = f"kmeans_k{k}_{key}_{kind}"
    
    cl = pd.Series(
            kmeans.labels_,
            index=labels,
            name=name
            )

    # store result
    target.clusters[name] = cl
    print(f'Cluster "{name}" stored! \n')
    
    print(f'Cluster results: \n{cl}\n')
    
    return target
