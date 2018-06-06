from sklearn.cluster import KMeans
import data_import as data


dataset = data.data().load()

y = dataset['species'].astype('category')
X = dataset.drop(['species'], axis=1)

k_means = KMeans(n_clusters=3)
k_means.fit(X)

print(k_means.labels_[::10])
print(y[::10])