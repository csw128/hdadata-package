from ..dynamic_models.representation import _get_vectors
from ..structure.object import HDAData
from sklearn.decomposition import PCA


def compute_PCA(
    data: HDAData,
    target: HDAData,
    key: str = None,
    kind: str = None,
    n_pc: int = None,
    trajectory_id: list = None,
    id: str = None
    ) -> HDAData:
    """Computes PCA to reduce dimensionality on the following variations:
        - gene x cell matrix (with gene expression values "X") -> cell x PC matrix
        - gene x cell matrix (with new RNA values) -> cell x PC matrix
        - trajectory x [gene x position] matrix (with gene expression values "X") -> trajectory x PC matrix 
        - trajectory x [gene x position] matrix (with new RNA values) -> trajectory x PC matrix     

        Args:
            data: 
                (subset) data object to analyze
            target: 
                location to store result - only for internal use by wrapper
            key: "X" / "ntr" / "new_rna" 
                values to compute on 
            kind: "gene" / "cell" / "trajectory"
                type of elements
            n_pc:
                number of Principle Components to reduce to
            trajectory_id: 
                selected trajectories to analyze
            id:
                name identifier of resulting reduced matrix
    
        Returns:
            reduced matrix stored in HDAData.layers
        
        """
    vectors, _ = _get_vectors(
                        data=data,
                        target=target,
                        key=key,
                        kind=kind,
                        trajectory_id=trajectory_id
                    )
    
    pca = PCA(
            n_components=n_pc,
            random_state=0
            )
        
    pca_matrix = pca.fit_transform(vectors)
    
    if id:
        name = f"pca_{kind}_{id}"
    else:
        name = f"pca_{kind}"
    
    # store result
    target.layers[name] = pca_matrix
    print(f'PCA matrix {kind} x {n_pc} PCs stored in HDAData.layers["{name}"] \n')

    return target