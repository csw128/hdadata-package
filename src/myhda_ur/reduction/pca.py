from sklearn.decomposition import PCA
import numpy as np

def reduce_pca(
    vectors: np.ndarray
    ) -> tuple[np.ndarray, str, str]:
    """Dimensionality reduction of vectors to 2D using PCA.
    
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)"
    ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)"
    
    return coords, xlabel, ylabel

