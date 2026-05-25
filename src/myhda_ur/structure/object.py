from typing import Optional, Dict, Union
import scipy.sparse as sp
import pandas as pd


class HDAData:
    def __init__(
        self, 
        X: sp.csr_matrix, 
        obs: Optional[pd.DataFrame] = None, 
        var: Optional[pd.DataFrame] = None, 
        layers: Optional[Dict[str, sp.csr_matrix]] = None, 
        similarities: Optional[Dict[str, pd.DataFrame]] = None,
        trajectories: Optional[pd.DataFrame] = None
        ) -> None:
        
        """Unified data structure for analaysis
        
        Args:
            X:              gene expression matrix (old/total RNA)
            obs:            cell metadata (barcodes)
            var:            gene metadata (features) 
            layers:         additional matrices (e.g layers["ntr"] (new RNA))
            similarities:   similarity matrices (e.g gene-gene matrix)
            trajectories:   dynamics over time
        
        """
        self.X = X                      
        self.obs = self._set_obs_idx(obs)                  
        self.var = self._set_var_idx(var)                  
        self.layers = layers or {}   
        self.similarities = similarities or {}   
        self.trajectories = trajectories   

        self._validate()
        self._create_idx_maps()
    

    def _validate(self) -> None:
        """Validate shape consistency between X, obs, and var (rows=genes, columns=cells). 
        
        Raises:
            ValueError: If X columns don't match obs length
            ValueError: If X rows don't match var length
            
        """
        if self.obs is not None and self.X.shape[1] != len(self.obs):
            raise ValueError(f"Mismatch: X has {self.X.shape[1]} columns (cells) but obs has {len(self.obs)} cells")

        if self.var is not None and self.X.shape[0] != len(self.var):
            raise ValueError(f"Mismatch: X has {self.X.shape[0]} rows (genes) but var has {len(self.var)} genes")
        
        
    @property
    def n_obs(self) -> int:
        """Number of observations (cells). """

        return self.X.shape[1]


    @property
    def n_vars(self) -> int:
        """Number of variables (genes). """
        
        return self.X.shape[0]
    
    
    def _set_obs_idx(
        self, 
        obs: Optional[pd.DataFrame]
        )-> Optional[pd.DataFrame]:
        """Set observation index to cell line names, and split information by dot to columns.
        
        For example: "A549.4h.AAAACGATCCTGGA"
        
        index | cell | time | barcode
        A549.4h.AAAACGATCCTGGA | A549 | 4h | AAAACGATCCTGGA
        
        """
        if obs is None:
            return None
        
        obs = obs.copy()
        
        names = obs.iloc[:, 0].astype(str)
        split = names.str.split(".", expand=True)
        
        obs["cell"] = split[0]
        obs["time"] = split[1]
        obs["barcode"] = split[2]
        
        # Set obs index to full cell names
        obs.index = names
        obs.index.name = "cell_name"
        
        obs.rename(columns={0: "name"}, inplace=True)
  
        return obs
    
    
    def _set_var_idx(
        self, 
        var: Optional[pd.DataFrame]
        ) -> Optional[pd.DataFrame]:
        """Set variable index to gene names, and split information by dot to columns.
        
        For example: "ENSG00000225602  MTOR-AS1  Gene Expression  Exonic (h.ens90)   705"
    
        index | gene id | gene symbol | feature type | annotation | length
        MTOR-AS1 | ENSG00000225602 | MTOR-AS1 | Gene Expression | Exonic (h.ens90) | 705
    
        """
        if var is None:
            return None
        
        var = var.copy()

        var.columns = [
            "gene_id",
            "gene_symbol",
            "feature_type",
            "annotation",
            "length"
        ]
        
        # Set var index to gene symbols
        var.index = var["gene_symbol"]     
        
        return var

    
    def _create_idx_maps(self) -> None:
        """Create bidirectional mapping between names and integer indices for lookup:
            1. cell name to index 
            2. index to cell names
            3. gene name to index
            4. index to gene name
        
        """
        if self.obs is not None:
            self._cell_to_index = {cell: idx for idx, cell in enumerate(self.obs.index)}
            self._index_to_cell = {idx: cell for idx, cell in enumerate(self.obs.index)}
        else:
            self._cell_to_index = {}
            self._index_to_cell = {}

        if self.var is not None:
            self._gene_to_index = {gene: idx for idx, gene in enumerate(self.var.index)}
            self._index_to_gene = {idx: gene for idx, gene in enumerate(self.var.index)}
        else:
            self._gene_to_index = {}
            self._index_to_gene = {}
    
    
    def __repr__(self) -> str:
        """String representation method of the HDAData object. """
        
        return (
            f"HDAData object with:\n"
            f"  n_vars (genes): {self.n_vars}\n"
            f"  n_obs (cells): {self.n_obs}\n"
            f"  X shape: {self.X.shape}\n"
            f"  layers: {list(self.layers.keys())}\n"
            f"  similarities: {list(self.similarities.keys())}\n"
            f"  trajectories: {'Yes' if self.trajectories is not None else 'No'}"
        )
        
        
    def show(
        self, 
        matrix: Optional[sp.csr_matrix] = None, 
        n_rows: int = None,
        n_cols: int = None,
        mode: str = "coo"
        ) -> None:
        """Display matrix in different formats with flexible row/column selection.
        
        Formats supported:
            1. COO Coordinate List format -> (row, col, value)  
            2. Dense format -> numeric matrix in numpy arrays
            3. Labelled view -> labelled rows and columns pandas DataFrame
    
        Arguments:
            matrix: matrix to display (e.g X by default, or ntr)
            n_rows: first n rows to display 
            n_cols: first n columns to display 
            mode: "coo", "dense", "label"
      
        """
        if matrix is None:
            matrix = self.X
        
        # Case 0: If no n_rows or n_cols specified, then show the full matrix
        if n_rows is None and n_cols is None:
            n_rows = matrix.shape[0]
            n_cols = matrix.shape[1]
            
        # Case 1: If matrix only has 1 row or 1 column
        if n_rows is None and matrix.shape[0] == 1:
            n_rows = 1
        if n_cols is None and matrix.shape[1] == 1:
            n_cols = 1
            
        # Case 2: Check that n_rows and n_cols doesn't exceed the matrix size
        if n_rows > matrix.shape[0] or n_cols > matrix.shape[1]:
            raise ValueError(f'(n_rows, n_cols) cannot be larger than {matrix.shape[0], matrix.shape[1]}!')
        
        # Mode 1: COO
        if mode == "coo":
            coo = matrix.tocoo()
            
            mask = (coo.row < n_rows) & (coo.col < n_cols)
            filtered_rows = coo.row[mask]
            filtered_cols = coo.col[mask]
            filtered_data = coo.data[mask]
            
            for i in range(len(filtered_data)):
                return (filtered_rows[i], filtered_cols[i], filtered_data[i])
        
        # Mode 2: Dense
        if mode == "dense":
            return matrix[:n_rows, :n_cols].toarray()
        
        # Mode 3: Labelled
        if mode == "label":
            sub = matrix[:n_rows, :n_cols].toarray()

            return pd.DataFrame(
                sub,
                index=self.var.index[:n_rows],     
                columns=self.obs.index[:n_cols]    
            )
                

    def _range_check(
        self, 
        ref: Union[str, int], 
        type=None
        ) -> int:
        """Check if cell or gene name exist, or cell or gene index is within range.
 
        Raises:
            KeyError: If name of cell not found 
            ValueError: If index out of range
                
        """
        if ref is None:
            return None
    
        if type=="cell":
            if isinstance(ref, str):
                if ref not in self._cell_to_index:
                    raise KeyError(f"Cell '{ref}' not found")
                return self._cell_to_index[ref] 
        
            else:
                if ref < 0 or ref >= self.n_obs:
                    raise ValueError(f"Cell index {ref} out of range")
                return ref
     
        if type=="gene":
            if isinstance(ref, str):
                if ref not in self._gene_to_index:
                    raise KeyError(f"Gene '{ref}' not found")
                return self._gene_to_index[ref]
                
            else:
                if ref < 0 or ref >= self.n_vars:
                    raise ValueError(f"Gene index {ref} out of range")
                return ref
            
    
    def get_expression(
        self, 
        gene: Union[str, int], 
        cell: Union[str, int]
        ) -> sp.csr_matrix:
        """Get expression value for specific gene and cell. 
        
        """
        gene_idx = self._range_check(gene, "gene")
        cell_idx = self._range_check(cell, "cell")
            
        return self.X[gene_idx, cell_idx]
    
    
    def _get_item_to_indices(
        self, 
        key: Union[str, int, list, slice, None], 
        type: str
        ) -> Union[str, int, list, slice, None]:
        """Convert string / slice / list to integer indices.
        
        Args:
            key: 
                input of cell / gene name or indices
            type: "gene" / "cell"
                item type whether gene or cell
            
        """
        if key is None or key == slice(None):
            return slice(None)
    
        # Case 1: single value
        if isinstance(key, (int, str)):
            return self._range_check(key, type)
        
        # Case 2: slice
        if isinstance(key, slice):
            start = self._range_check(key.start, type) if key.start is not None else None
            stop = self._range_check(key.stop, type) if key.stop is not None else None
            return slice(start, stop, key.step)
        
        # Case 3: list/array
        if isinstance(key, (list, tuple)):
            return [self._range_check(i, type) for i in key]
        
    
    def _subset_metadata(
        self, 
        data: Optional[pd.DataFrame], 
        key: Union[str, int, list, slice, None]
        ) -> Optional[pd.DataFrame]:
        """Subset metadata df.
        
        """
        if data is None:
            return None
        
        if key == slice(None):
            return data
        
        if isinstance(key, int):
            return data.iloc[[key]]
        
        if isinstance(key, list):
            return data.iloc[key]
        
        if isinstance(key, slice):
            return data.iloc[key]
        
        return None
    
    
    def __getitem__(self, key: tuple) -> 'HDAData':
        """Slice or get subset of HDAData using gene or cell names or index.
    
        Args:
            key: Tuple of (gene, cell) where each can be a slice, list, or integer index
                    
        Examples:
            data['Gene1', 'Cell1']                      # Single gene and cell
            data[['Gene1', 'Gene2'], :]                 # Multiple genes, all cells
            data[:, ['Cell1', 'Cell2']]                 # All genes, specific cells
            data['Gene1':'Gene10', 'Cell1':'Cell5']     # Slice by name
            data[:5, ['Cell1', 'Cell2']]                # Slice by gene index and cell names
            data[['Gene1', 'Gene2'], :5]                # Slice by gene name and cell index
            data[10:20, 30:40]                          # Integer indexing
    
        Returns:
            A new HDAData object containing the subset 
            
        """
        gene_k, cell_k = key
        
        var_idx = self._get_item_to_indices(gene_k, "gene")
        obs_idx = self._get_item_to_indices(cell_k, "cell")
        
        # Subset X
        if isinstance(var_idx, slice) and var_idx == slice(None):
            X_sub = self.X[:, obs_idx]
        else:
            X_sub = self.X[var_idx, :][:, obs_idx]
        
        # Subset metadata
        var_sub = self._subset_metadata(self.var, var_idx)
        obs_sub = self._subset_metadata(self.obs, obs_idx)

        # Subset layers
        layers_sub = {
            k: (v[:, obs_idx] if isinstance(var_idx, slice) and var_idx == slice(None) 
                else v[var_idx, :][:, obs_idx])
            for k, v in self.layers.items()
        }

        return HDAData(
            X_sub, 
            obs=obs_sub, 
            var=var_sub, 
            layers=layers_sub, 
            trajectories=self.trajectories
        )
    
    
    def compute_similarity(
        self,
        data,
        key: str,
        kind: str,
        trajectory_id: list = None,
        method: str = None,
        name: str = None
        ):
        """Wrapper method to call compute_similarity() from HDAData object.
     
        """
        from ..dynamic_models.similarity import compute_similarity as _compute_similarity
        return _compute_similarity(
            data=data,
            target=self,  
            key=key,
            kind=kind,
            trajectory_id=trajectory_id,
            method=method,
            name=name
        )


    def get_highly_variable_genes(
        self,
        n: int
        ) -> list:
        """Wrapper method to call get_highly_variable_genes() from HDAData object.
    
        """
        from ..dynamic_models.selector import get_highly_variable_genes as _get_highly_variable_genes
        return _get_highly_variable_genes(
            data=self,
            n=n
            )
        
        
    def get_timepoint_cells(
        self,
        n: int
        ) -> list:
        """Wrapper method to call get_timepoint_cells() from HDAData object.
    
        """
        from ..dynamic_models.selector import get_timepoint_cells as _get_timepoint_cells
        return _get_timepoint_cells(
            data=self,
            n=n
            )
        
        
    def get_highly_dynamic_trajectories(
        self,
        n: int
        ) -> list:
        """Wrapper method to call get_highly_dynamic_trajectories() from HDAData object.
    
        """
        from ..dynamic_models.selector import get_highly_dynamic_trajectories as _get_highly_dynamic_trajectories
        return _get_highly_dynamic_trajectories(
            data=self,
            n=n
            )
    