from scipy.io import mmread
import gzip
import pandas as pd


def read_mtx(path: str):
    '''
    Reads the Matrix Market (.mtx or .mtx.gz)) file and returns a sparse matrix. 
    The sparse matrix is converted to Compressed Sparse Row (csr) format.
    '''
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return mmread(f).tocsr()       
    return mmread(path).tocsr()

def read_tsv(path: str):
    return pd.read_csv(path, sep="\t", header=None)

def read_csv(path: str):
    return pd.read_csv(path)
