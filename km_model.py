import data_preprocessor
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

class KMModel:

    coordinates = data_preprocessor.DataPreprocessor().transform_to_coordinates()

    def __init__(self, coordinates):

        self.coordinates = coordinates
        self.kmeans = KMeans()

    def fit_elbow(self):

        WCSS = []

        for k in range(1, 17):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1)
            kmeans.fit(self.coordinates)
            WCSS.append(kmeans.inertia_)

        return WCSS

    def fit_shiloette(self):

        score_list = {}
        for k in range(2, 18):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1).fit(self.coordinates).labels_
            score = silhouette_score(self.coordinates, kmeans, metric='euclidean')
            score_list[k] = score

        number_of_clusters = max(score_list, key=lambda key: score_list[key])

        return number_of_clusters

### need to be fixed to work with plotters.py methods
    def fit_predict(self, n_clusters):

        kmeans = KMeans(n_clusters=n_clusters, n_init=300, random_state=1)
        kmeans.fit_predict(self.coordinates.values)
        # labels = kmeans.labels_
        # centroids = kmeans.cluster_centers_
        h = 0.001
        x_min, x_max = self.coordinates['longitude'].min(), self.coordinates['longitude'].max()
        y_min, y_max = self.coordinates['latitude'].min(), self.coordinates['latitude'].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return

# coordinates = KMModel.coordinates
# model = KMModel(coordinates)
#
# number_of_clusters = model.fit_shiloette()
# print(number_of_clusters)
#
# kmeans = KMeans(n_clusters=number_of_clusters, n_init=300, random_state=1)
# kmeans.fit_predict(coordinates.values)
#
#
# h = 0.001
# x_min, x_max = coordinates['longitude'].min(), coordinates['longitude'].max()
# y_min, y_max = coordinates['latitude'].min(), coordinates['latitude'].max()
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# plt.figure(1, figsize=(10,4))
# plt.clf()
# plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Pastel1, origin='lower')
# plt.scatter(x=coordinates['longitude'], y=coordinates['latitude'], c=labels, s=100)
# plt.scatter(x=centroids[:, 0], y=centroids[:, 1], s=300, c='red')
# plt.xlabel('Longitude (x)')
# plt.ylabel('Latitude (y)')
# plt.title('Clustering')
# plt.show()
