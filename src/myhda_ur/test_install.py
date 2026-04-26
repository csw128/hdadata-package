# def hello():
#     print("Hello!")


from myhda_ur import read_mtx, HDAData

X = read_mtx("data/matrix.mtx.gz")
data = HDAData(X)

print(data.n_obs)