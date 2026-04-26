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

# print(data.X)
# print(data.show_nonzero(10))

# data_subset = data[[0, 5], [10, 20]]   # genes: 0, 5, cells: 10, 20
# print(data_subset)
# rows, cols = data_subset.X.nonzero()
# print(data_subset.X.nnz)    # check number of non-zero values (if 0, means matrix is all zeroes)
# print(data_subset.X.toarray()) 