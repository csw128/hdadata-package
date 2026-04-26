class HDAData:
    def __init__(self, X, obs=None, var=None, layers=None, trajectories=None):
        self.X = X                      # gene expression matrix -> total RNA
        self.obs = obs                  # cell metadata (barcodes)
        self.var = var                  # gene metadata (features)  -> layers["ntr"] = new RNA
        self.layers = layers or {}      # additional matrices (e.g. NTR)
        self.trajectories = trajectories   # dynamics

        self._validate()

    def _validate(self):
        '''
        Checks shape consistency (rows=genes, columns=cells)
        '''
        if self.obs is not None and self.X.shape[1] != len(self.obs):
            raise ValueError(f"Mismatch: X has {self.X.shape[1]} columns but obs has {len(self.obs)} cells")

        if self.var is not None and self.X.shape[0] != len(self.var):
            raise ValueError(f"Mismatch: X has {self.X.shape[0]} rows but var has {len(self.var)} genes")
        
    @property
    def n_obs(self):
        '''
        Returns number of observations (cells)
        '''
        return self.X.shape[1]

    @property
    def n_vars(self):
        '''
        Returns of variables (genes)
        '''
        return self.X.shape[0]
    
    def __getitem__(self, index):
        '''
        Slicing matrix
        '''
        obs_idx, var_idx = index

        X = self.X[obs_idx][:, var_idx]

        obs = self.obs.iloc[obs_idx] if self.obs is not None else None
        var = self.var.iloc[var_idx] if self.var is not None else None

        layers = {
            k: v[obs_idx][:, var_idx]
            for k, v in self.layers.items()
        }

        return HDAData(X, obs=obs, var=var, layers=layers, trajectories=self.trajectories)
    
    def __repr__(self):
        '''
        String representation method for object 
        '''
        return (
            f"HDAData object with:\n"
            f"  n_obs (cells): {self.n_obs}\n"
            f"  n_vars (genes): {self.n_vars}\n"
            f"  X shape: {self.X.shape}\n"
            f"  layers: {list(self.layers.keys())}\n"
            f"  trajectories: {'Yes' if self.trajectories is not None else 'No'}"
        )
        
    def show_nonzero(self, n=5):
        '''
        Cleaner representation of data in form of (row, col, value)
        '''
        coo = self.X.tocoo()
        for i in range(min(n, len(coo.data))):
            print(coo.row[i], coo.col[i], coo.data[i])