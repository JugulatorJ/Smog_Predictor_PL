import warnings
from km_model import KMModel

warnings.filterwarnings("ignore")


def main():

    coordinates = KMModel.coordinates
    model = KMModel(coordinates)
    print('Do you want to plot locations?')
    answer = input('y/n: ')
    if answer == 'y':
        model.plot_loc_data()
    else:
        pass
    print('Do you want to plot elbow chart and choose number of clusters manually? If not program will use silhouette score')
    if input('y/n: ') == 'y':
        model.plot_wcss_elbow()
        number_of_clusters = int(input('Choose number of clusters: '))
        model.clustering(number_of_clusters)
    else:
        number_of_clusters = model.fit_shiloette()
        clustered_model = model.clustering(number_of_clusters)
        model.plot_clusters(clustered_model)
        user_cluster = model.user_cluster(clustered_model)
        model.plot_user_cluster(clustered_model, user_cluster)

if __name__ == '__main__':
    main()