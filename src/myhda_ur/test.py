from read.load import load_files
from structure.object import HDAData

loaded = load_files(
    gene_expression_matrix="../../data/matrix.mtx.gz",
    barcodes_data="../../data/barcodes.tsv.gz",
    features_data="../../data/features.tsv.gz",
    new_to_total_ratio_matrix="../../data/4sU.Binom.ntr.mtx.gz",
    trajectory_table="../../data/trajectories_sci.csv"
)

data = HDAData(
    X=loaded["gene_expression_matrix"],
    obs=loaded["barcodes_data"],
    var=loaded["features_data"],
    layers={"ntr": loaded["new_to_total_ratio_matrix"]},
    trajectories=loaded["trajectory_table"]
)


# print(data)
# print(data.X)

# print(data.obs.head())
# print("-------------------------")
# print(data.var.head())


# # Test get 
# print(data.get_cell("A549.10h.GAGCTGAGACTCTA"))
# print(data.get_cell(1))

# print(data.get_gene("MTOR-AS1"))
# print(data.get_gene(1))

# print("Show gene expression of gene MTOR-AS1 and cell A549.10h.GAGCTGAGACTCTA")
# print(f'{data.get_expression("MTOR-AS1", "A549.10h.GAGCTGAGACTCTA")}\n')


# # Test various indexing patterns
# single_gene = data["MTOR-AS1", :]  
# print(single_gene)
# print(single_gene.X)

# single_value = data["MTOR-AS1", "A549.10h.GAGCTGAGACTCTA"]  
# print(single_value)

# print("Show all cells for these genes: MTOR-AS1, ANGPTL7")
# gene_list = data[["MTOR-AS1", "ANGPTL7"], :]
# print(f'{gene_list}\n')
# print(gene_list.X)

# cell_list = data[:, ["A549.10h.GAGCTGAGACTCTA", "A549.0h.GGTCACGGCAACCA"]]
# print(cell_list)

# gene_slice = data["MTOR-AS1":"ANGPTL7", :]
# print(gene_slice)

# mixed_index = data[["MTOR-AS1", "ANGPTL7"], 10:20]
# print(mixed_index)



# subset = data[10:20, 30:40]
# print(subset.X)
# print("-------------------------")
# print(subset.show(subset.X, n=5, mode="coo"))
# print("-------------------------")
# print(subset.show(subset.X, n=10, mode="dense"))
# print("-------------------------")
# print(subset.show(subset.X, n=5, mode="label"))