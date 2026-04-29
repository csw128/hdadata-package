from read.load import read_mtx, read_tsv, read_csv
from structure.object import HDAData

X = read_mtx("../../data/matrix.mtx.gz")
ntr = read_mtx("../../data/4sU.Binom.ntr.mtx.gz")

obs = read_tsv("../../data/barcodes.tsv.gz")
var = read_tsv("../../data/features.tsv.gz")

traj = read_csv("../../data/trajectories_sci.csv")

data = HDAData(
    X,
    obs=obs,
    var=var,
    layers={"ntr": ntr},
    trajectories=traj
)


# print(data)
# print(data.X)
# print(data.show(10))

# print(data.get_cell("A549.10h.GAGCTGAGACTCTA"))
# print(data.get_cell(1))

# print(data.get_gene("MTOR-AS1"))
# print(data.get_gene(1))

# print(data.get_expression("MTOR-AS1", "A549.10h.GAGCTGAGACTCTA"))



# # Test various indexing patterns
# single_gene = data["MTOR-AS1", :]  
# print(single_gene)
# print(single_gene.X)

# single_value = data["MTOR-AS1", "A549.10h.GAGCTGAGACTCTA"]  
# print(single_value)

# gene_list = data[["MTOR-AS1", "ANGPTL7"], :]
# print(gene_list)

# cell_list = data[:, ["A549.10h.GAGCTGAGACTCTA", "A549.0h.GGTCACGGCAACCA"]]
# print(cell_list)

# gene_slice = data["MTOR-AS1":"ANGPTL7", :]
# print(gene_slice)

# mixed_index = data[["MTOR-AS1", "ANGPTL7"], 10:20]
# print(mixed_index)




#################

# rows, cols = data_subset.X.nonzero()
# print(data_subset.X.nnz)    # check number of non-zero values (if 0, means matrix is all zeroes)
# print(data_subset.X.toarray()) 