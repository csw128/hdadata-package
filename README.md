# HDAData package documentation

This package serves as a high-dimensional data analysis tool for Python using the sci-fate data, as part of the DAT-M-HIGHDIM lab project.

## How to install
1. Make sure you are in the root folder directory where `pyproject.toml` lives. 

2. In the `data` folder in the root directory, place the five sci-fate dataset files:
    - 4sU.Binom.ntr.mtx.gz
    - barcodes.tsv.gz
    - features.tsv.gz
    - matrix.mtx.gz
    - trajectories_sci.csv

3. In your terminal, install the package by running the following command:

    ```bash
    pip install -e .
    ```


## References & Resources
[Sci-fate paper](https://www.nature.com/articles/s41587-020-0480-9)

[Sci-fate preprocessed data](https://zenodo.org/records/14176698?preview_file=SciFate.zip)