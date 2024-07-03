from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=200)

plt.scatter(x[:, 0], x[:, 1])
plt.show()
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,10), inertia)
plt.xlabel("Broj klaster")
plt.ylabel("Inercija")
plt.show()

clusters = int(input())


model = KMeans(n_clusters=clusters)
model.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1])
plt.scatter(model.cluster_centers_[:,0], 
model.cluster_centers_[:,1], s=300, c='red')
plt.show()
