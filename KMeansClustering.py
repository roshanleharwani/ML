from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

data = load_digits()

data = scale(load_digits.data)# To scale the data between 0 to 1

model = KMeans(n_clusters=10,init='random',n_init=10)
model.fit(data)


