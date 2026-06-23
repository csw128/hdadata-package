from ..structure.object import HDAData
from ..dynamic_models.representation import _get_vectors
import scipy.sparse as sp
import umap


def reduce_UMAP(
    data: HDAData = None,
    target: HDAData = None,
    key: str = None,
    kind: str = None,
    trajectory_id: list = None,
    id: str = None
    ) -> HDAData:
    
    if key == "pca":
        if "pca" not in target.layers:
            raise ValueError('PCA matrix not found in HDAData.layers["pca"] '
                             'Run compute_PCA first.')
        vectors = target.layers["pca"]          
    
    else:
        vectors, _ = _get_vectors(
                            data=data,
                            key=key,
                            kind=kind,
                            trajectory_id=trajectory_id
                        )
    
    reducer = umap.UMAP(
                    n_components=2, 
                    random_state=0
                    )
    
    coords = reducer.fit_transform(vectors)
    xlabel, ylabel = "UMAP1", "UMAP2"

    em = (coords, xlabel, ylabel)

    name = f"umap_{kind}_{id}"

    target.embeddings[name] = em
    print(f'UMAP embedding "{name}" stored! \n')

    return target