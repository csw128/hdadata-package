from structure.object import HDAData
import numpy as np


def get_highly_variable_genes(
    data,
    n: int
    ):
    """ 
    Identify highly variable genes by computing the NTR variance across cells 
    for each gene. Then identify genes with the highest variances.
    
    """
    dense = data.layers["ntr"].toarray()
    gene_var = np.var(dense, axis=1)
    top_idx = np.argsort(gene_var)[::-1][:n]
    selected_genes = [data._index_to_gene[idx] for idx in top_idx]
    
    return selected_genes

