import gzip
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from typing import Dict


def read_mtx(path: str) -> sp.csr_matrix:
    """Read the Matrix Market (.mtx or .mtx.gz)) file as a sparse matrix. 
       The sparse matrix is converted to Compressed Sparse Row (csr) format.
        
        Args:
            path: path to the input .mtx or .mtx.gz file
    
        Returns:
            sparse matrix in csr format
            
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return mmread(f).tocsr()       
    return mmread(path).tocsr()


def read_tsv(path: str) -> pd.DataFrame:
    """Read .tsv file as pandas Dataframe. 
        
        Args:
            path: path to the input .tsv file
    
        Returns:
            dataframe with the TSV contents       
            
    """
    return pd.read_csv(path, sep="\t", header=None)


def read_csv(path: str) -> pd.DataFrame:
    """Read .csv file as a pd Dataframe. 
        
        Args:
            path: path to the input .csv file
    
        Returns:
            dataframe with the CSV contents  
            
    """
    return pd.read_csv(path)


def load_files(
    gene_expression_matrix: str = None,
    barcodes_data: str = None,
    features_data: str = None,
    new_to_total_ratio_matrix: str = None,
    trajectory_table: str = None
    ) -> Dict: 
    """Load all the data files with a single function.
        
        Args:
            Paths to the following files: 
                - gene_expression_matrix
                - barcodes_data
                - features_data
                - new_to_total_ratio_matrix
                - trajectory_table 
    
        Returns:
            dictionary with file data loaded as variables.  
            
    """
    required = {
        "X_path": gene_expression_matrix,
        "obs_path": barcodes_data,
        "var_path": features_data,
        "ntr_path": new_to_total_ratio_matrix,
        "traj_path": trajectory_table
    }
    
    for name, path in required.items():
        if path is None:
            raise ValueError(f"{name} file path must be provided!")
    
    X = read_mtx(gene_expression_matrix)
    obs = read_tsv(barcodes_data)
    var = read_tsv(features_data)
    ntr = read_mtx(new_to_total_ratio_matrix)
    traj = read_csv(trajectory_table)

    return {
        "X": X,
        "obs": obs,
        "var": var,
        "ntr": ntr,
        "traj": traj
    }