import data_preprocessor
import km_model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time


class Plotter:

    def __init__(self):

        pass

    @staticmethod
    def plot_loc_data():

        coordinates = data_preprocessor.DataPreprocessor().transform_to_coordinates()
        plt.scatter(coordinates['longitude'], coordinates['latitude'])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Smog Data Locations')
        plt.savefig(rf'C:\Users\HP\PycharmProjects\Smog_Predictor_PL\plots\smog_locs{str(int(time()))}.png',bbox_inches='tight')
        plt.show()

        return

    @staticmethod
    def plot_wcss_elbow():

        wcss = km_model.KMModel(km_model.KMModel.coordinates).fit_elbow()
        plt.plot(1, 17, wcss, c='magenta')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.savefig(rf'C:\Users\HP\PycharmProjects\Smog_Predictor_PL\plots\elbow{str(int(time()))}.png',bbox_inches='tight')
        plt.show()

        return

### Needs to be fixed to be able to plot clusters like in scratch.py
    # @staticmethod
    # def plot_clusters():
    #     coordinates = data_preprocessor.DataPreprocessor().transform_to_coordinates()
    #     kmeans = km_model.KMModel(km_model.KMModel.coordinates).fit_predict(int(input('Enter number of clusters: ')))
    #     KM = KMeans()
    #     labels = KM.labels_
    #     centroids = KM.cluster_centers_
    #     plt.figure(1, figsize=(10, 4))
    #     plt.clf()
    #     plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #                cmap=plt.cm.Pastel1, origin='lower')
    #     plt.scatter(x=coordinates['longitude'], y=coordinates['latitude'], c=labels, s=100)
    #     plt.scatter(x=centroids[:, 0], y=centroids[:, 1], s=300, c='red')
    #     plt.xlabel('Longitude (x)')
    #     plt.ylabel('Latitude (y)')
    #     plt.title('Clustering')
    #     plt.show()

# Plotter().plot_loc_data()
# Plotter().plot_wcss_elbow()

