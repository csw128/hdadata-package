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
        
    target.layers[name] = pca_matrix
    print(f'PCA matrix {kind} x {n_pc} PCs stored in HDAData.layers["{name}"] \n')

    return target