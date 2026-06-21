from ..dynamic_models.representation import _get_vectors
from ..structure.object import HDAData
from sklearn.decomposition import PCA
import numpy as np


def reduce_PCA(
    data: HDAData,   
    target: HDAData,
    key: str = None,
    kind: str = None,
    trajectory_id: list = None,
    ) -> tuple[np.ndarray, str, str]:
    """Dimensionality reduction using PCA on raw gene/cell/trajectory vectors.
    
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
        
        Return:
            tuple:
                - x, y coordinates in reduced 2D PCA
                - xlabel: PCA1
                - ylabel: PCA2
                
    """
    vectors, _ = _get_vectors(
                        data=data,
                        key=key,
                        kind=kind,
                        trajectory_id=trajectory_id
                    )
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    xlabel, ylabel = "PC1", "PC2"
    
    em = (coords, xlabel, ylabel)
    
    name = f"pca_{kind}"
     
    # store result
    target.embeddings[name] = em
    print(f'PCA embedding "{name}" stored! \n')

    return target


def compute_PCA(
    data: HDAData,
    target: HDAData,
    key: str = None,
    kind: str = None,
    n_pc: int = None,
    trajectory_id: list = None,
    ) -> HDAData:
    
    vectors, _ = _get_vectors(
                        data=data,
                        key=key,
                        kind=kind,
                        trajectory_id=trajectory_id
                    )
    
    pca = PCA(n_components=n_pc)
    pca_matrix = pca.fit_transform(vectors)
    
    # store pc matrix 
    target.layers["pca"] = pca_matrix
    print(f'PCA matrix {key} x {n_pc} PCs stored in data.layers["pca"] \n')
   
    # store first 2 PCs coordinates for plotting 
    coords = pca_matrix[:, :2]      
    xlabel, ylabel = "PC1", "PC2"
    
    em = (coords, xlabel, ylabel)
    
    name = f"pca_{kind}_PC{n_pc}"
     
    # store result
    target.embeddings[name] = em
    print(f'PCA embedding "{name}" stored! \n')

    return target