# HDAData

**HDAData** is a Python package for high-dimensional single-cell data analysis, built around the `HDAData` object. It is developed as part of the DAT-M-HIGHDIM lab project and is demonstrated on the [Sci-fate](https://www.nature.com/articles/s41587-020-0480-9) dataset, with support for gene / cell / trajectory similarity analysis, clustering, dimensionality reduction, and visualization.

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Getting started](#getting-started)
- [References](#references)

## Requirements

- **Python >= 3.9**
- [Sci-fate dataset](https://zenodo.org/records/14176698?preview_file=SciFate.zip) (see [Installation](#installation) step 3)

## Installation

1. **Unzip the package** and open the `hdadata` folder in your preferred IDE (e.g. VS Code). All following commands should be run from this root directory, where `pyproject.toml` lives.

2. **(Optional) Create a dedicated environment.** Recommended if you don't want the package's dependencies installed into your base Python environment:
    ```bash
    conda create --name hdadata python=3.10
    conda activate hdadata
    conda install ipykernel
    ```
    Installing `ipykernel` lets you select this environment as a kernel in Jupyter Notebook later.

3. **Download the Sci-fate dataset** from [Zenodo](https://zenodo.org/records/14176698?preview_file=SciFate.zip) if you do not already have it. Place the following five files into the `data/` folder in the root:
    - `4sU.Binom.ntr.mtx.gz`
    - `barcodes.tsv.gz`
    - `features.tsv.gz`
    - `matrix.mtx.gz`
    - `trajectories_sci.csv`

4. **Install the package** and its dependencies:
    ```bash
    pip install -e .
    ```
    This installs all required dependencies directly from `pyproject.toml`. No separate `requirements.txt` install step is needed.

5. **Confirm the package is installed** when you see the following message in your terminal:
    ```bash
    Successfully built hdadata
    ```

## Getting started

Open `Tutorial.ipynb` for a full walkthrough of the package, including loading data, computing similarities, clustering, and dimensionality reduction.

If you are using a **conda environment**, then in your Jupyter notebook, make sure to select the **kernel** matching the environment you created in [Installation](#installation) step 2 before running the notebook.

A minimal example on how to import the package after successful installation:

```python
import hdadata as hda

hdata = hda.load_scifate()
```

For further information, please see `Tutorial.ipynb`.

## References

- [Sci-fate original paper](https://www.nature.com/articles/s41587-020-0480-9)
- [Sci-fate dataset from Zenodo](https://zenodo.org/records/14176698?preview_file=SciFate.zip)
- [AnnData Documentation](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html)
- [Scanpy Documentation](https://scanpy.readthedocs.io/en/stable/api/index.html)
- [Seurat Documentation](https://satijalab.org/seurat/articles/pbmc3k_tutorial.html)
