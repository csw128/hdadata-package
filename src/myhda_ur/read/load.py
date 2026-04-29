import gzip
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from typing import Optional


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
