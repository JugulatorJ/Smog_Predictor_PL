import warnings
from km_model import KMModel
from data_preprocessor import DataPreprocessor
warnings.filterwarnings("ignore")


def main():

    coordinates = DataPreprocessor().transform_to_coordinates()
    model_km = KMModel(coordinates)

    while True:

        print('Do you want to plot locations?')

        answer = input('y/n: ')

        if answer.lower() == 'y':
            model_km.plot_loc_data()
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
            model_km.plot_wcss_elbow()
            while True:
                try:
                    number_of_clusters = int(input('Choose number of clusters: '))
                    break
                except ValueError:
                    print('Wrong value. Provide integer number. Try again!')
                    continue
            clustered_model = model_km.clustering(number_of_clusters)
            model_km.plot_clusters(clustered_model)
            user_cluster = model_km.user_cluster(clustered_model)
            model_km.plot_user_cluster(clustered_model, user_cluster)
            break

        elif answer.lower() == 'n':
            number_of_clusters = model_km.fit_shiloette()
            clustered_model = model_km.clustering(number_of_clusters)
            model_km.plot_clusters(clustered_model)
            user_cluster = model_km.user_cluster(clustered_model)
            model_km.plot_user_cluster(clustered_model, user_cluster)
            break

        else:
            print("Wrong value. Try again!")


if __name__ == '__main__':
    main()