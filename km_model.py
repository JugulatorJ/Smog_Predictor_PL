import data_preprocessor
import locations
import numpy as np
from time import time
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


class KMModel:

    coordinates = data_preprocessor.DataPreprocessor().transform_to_coordinates()
    user_loc = locations.UserLoc().get_user_loc()

    def __init__(self, coordinates):

        self.coordinates = coordinates
        self.kmeans = KMeans()

    def fit_elbow(self):

        wcss = []

        for k in range(1, 17):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1)
            kmeans.fit(self.coordinates)
            wcss.append(kmeans.inertia_)

        return wcss

    def fit_shiloette(self):

        score_list = {}
        for k in range(2, 18):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1).fit(self.coordinates).labels_
            score = silhouette_score(self.coordinates, kmeans, metric='euclidean')
            score_list[k] = score

        number_of_clusters = max(score_list, key=lambda key: score_list[key])

        return number_of_clusters

# need to be fixed to work with plotters.py methods

    def clustering(self, number_of_clusters):
        kmeans = KMeans(n_clusters=number_of_clusters, n_init=300, random_state=1)
        kmeans.fit_predict(self.coordinates.values)
        h = 0.001
        x_min, x_max = self.coordinates['longitude'].min(), self.coordinates['longitude'].max()
        y_min, y_max = self.coordinates['latitude'].min(), self.coordinates['latitude'].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        plt.figure(1, figsize=(10, 4))
        plt.clf()
        plt.imshow(z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Pastel1, origin='lower')
        plt.scatter(x=self.coordinates['longitude'], y=self.coordinates['latitude'], c=labels, s=100)
        plt.scatter(x=centroids[:, 0], y=centroids[:, 1], s=300, c='red')
        plt.scatter(x=self.user_loc['longitude'], y=self.user_loc['latitude'], s=300, c='orange', marker='x')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')
        plt.title('Clustering')
        plt.savefig(rf'C:\Users\HP\PycharmProjects\Smog_Predictor_PL\plots\clusters{str(int(time()))}.png',
                    bbox_inches='tight')
        plt.show()

        return

    def plot_loc_data(self):

        plt.scatter(self.coordinates['longitude'], self.coordinates['latitude'])
        plt.scatter(x=self.user_loc['longitude'], y=self.user_loc['latitude'], s=300, c='orange', marker='x')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Smog Data Locations')
        plt.savefig(rf'C:\Users\HP\PycharmProjects\Smog_Predictor_PL\plots\smog_locs{str(int(time()))}.png',
                    bbox_inches='tight')
        plt.show()

        return

    def plot_wcss_elbow(self):
        wcss = KMModel(self.coordinates).fit_elbow()
        plt.plot(1, 17, wcss, c='magenta')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.savefig(rf'C:\Users\HP\PycharmProjects\Smog_Predictor_PL\plots\elbow{str(int(time()))}.png',
                    bbox_inches='tight')
        plt.show()

        return




