import gzip
from pathlib import Path
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from typing import Dict, Optional, Union
from ..structure.object import HDAData


class Loader:
    def __init__(
        self,
        dir: Optional[Union[str, Path]] = None
        ):
        """Data Loader for files into HDAData object.
        
        Args: 
            dir: default /data directory or user specificed folder path
        """
        self.dir = Path(__file__).parent.parent.parent.parent / "data"
    
    
    @staticmethod
    def read_mtx(path: Union[str, Path]) -> sp.csr_matrix:
        """Read the Matrix Market (.mtx or .mtx.gz)) file as a sparse matrix. 
        The sparse matrix is converted to Compressed Sparse Row (csr) format.
        
        Args:
            path: path to the input .mtx or .mtx.gz file
    
        Returns:
            sparse matrix in csr format
                
        """
        path = Path(path)
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                return mmread(f).tocsr()       
        return mmread(path).tocsr()


    @staticmethod
    def read_tsv(path: Union[str, Path]) -> pd.DataFrame:
        """Read .tsv file as pandas Dataframe. 
            
        Args:
            path: path to the input .tsv file
    
        Returns:
            dataframe with the TSV contents       
                
        """
        path = Path(path)
        compression = 'gzip' if path.suffix == '.gz' else None
        return pd.read_csv(path, sep='\t', header=None, compression=compression)


    @staticmethod
    def read_csv(path: Union[str, Path]) -> pd.DataFrame:
        """Read .csv file as a pd Dataframe. 
            
        Args:
            path: path to the input .csv file
    
        Returns:
            dataframe with the CSV contents  
                
        """
        return pd.read_csv(path, index_col=0)


    def get_file_path(
        self, 
        key: str, 
        ) -> Path:
        """Resolve file path
        
        Args:
            key: file identifier
                
        """
        default_map = {
            'gene_expression_matrix': 'matrix.mtx.gz',
            'barcodes_data': 'barcodes.tsv.gz',
            'features_data': 'features.tsv.gz',
            'new_to_total_ratio_matrix': '4sU.Binom.ntr.mtx.gz',
            'trajectory_table': 'trajectories_sci.csv'
        }
        
        if key in default_map:
            return self.dir / default_map[key]

    
    def load_files(
        self,
        ) -> Dict: 
        """Load all the data files.
            
        Returns:
            dictionary with file data loaded 
                
        """
        required = ['gene_expression_matrix', 'barcodes_data', 'features_data']
        
        paths = {
            'gene_expression_matrix': self.get_file_path('gene_expression_matrix'),
            'barcodes_data': self.get_file_path('barcodes_data'),
            'features_data': self.get_file_path('features_data'),
            'new_to_total_ratio_matrix': self.get_file_path('new_to_total_ratio_matrix'),
            'trajectory_table': self.get_file_path('trajectory_table')
        }
        
        for f in required:
            if not paths[f].exists():
                raise FileNotFoundError(f"{f} file path must be provided!")
        
        X = self.read_mtx(paths['gene_expression_matrix'])
        obs = self.read_tsv(paths['barcodes_data'])
        var = self.read_tsv(paths['features_data'])
        
        if paths['new_to_total_ratio_matrix'].exists():
            ntr = self.read_mtx(paths['new_to_total_ratio_matrix'])
        else:
            None
        
        if paths['trajectory_table'].exists():
            traj = self.read_csv(paths['trajectory_table'])
        else:
            None

        return {
            "gene_expression_matrix": X,
            "barcodes_data": obs,
            "features_data": var,
            "new_to_total_ratio_matrix": ntr,
            "trajectory_table": traj
        }
       
        
    def load_as_hda(self) -> HDAData:
        """Load file data directly into HDAData object.
        
        Returns:
            HDAData Object
            
        """
        loaded = self.load_files()
        
        return HDAData(
            X=loaded["gene_expression_matrix"],
            obs=loaded["barcodes_data"],
            var=loaded["features_data"],
            layers={"ntr": loaded["new_to_total_ratio_matrix"]},
            trajectories=loaded["trajectory_table"]
        )