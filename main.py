import warnings
from km_model import KMModel

warnings.filterwarnings("ignore")


def main():

    coordinates = KMModel.coordinates
    clean_data = KMModel.cleaned_data
    model = KMModel(coordinates)

    while True:

        print('Do you want to plot locations?')

        answer = input('y/n: ')

        if answer.lower() == 'y':
            model.plot_loc_data()
            break

        elif answer.lower() == 'n':
            print('Proceeding.')
            break

        else:
            print("Wrong value. Try again!")

    while True:

        print('Do you want to plot elbow chart and choose number of clusters manually? '
              'If not program will use silhouette score')
        answer = input('y/n: ')

        if answer.lower() == 'y':
            model.plot_wcss_elbow()
            number_of_clusters = int(input('Choose number of clusters: '))
            clustered_model = model.clustering(number_of_clusters)
            model.plot_clusters(clustered_model)
            user_cluster = model.user_cluster(clustered_model)
            model.plot_user_cluster(clustered_model, user_cluster)
            break

        elif answer.lower() == 'n':
            number_of_clusters = model.fit_shiloette()
            clustered_model = model.clustering(number_of_clusters)
            model.plot_clusters(clustered_model)
            user_cluster = model.user_cluster(clustered_model)
            model.plot_user_cluster(clustered_model, user_cluster)
            break

        else:
            print("Wrong value. Try again!")


if __name__ == '__main__':
    main()