import json
import pandas as pd
import requests


class DataPackage:

    def __init__(self):

        pass

    @staticmethod
    def get_smog_data():

        smog_ds = json.loads(requests.get('https://public-esa.ose.gov.pl/api/v1/smog').text)
        size_of_smog_ds = len(smog_ds['smog_data'])

        if size_of_smog_ds <= 1200:
            print(f'Length of currently available dataset is {size_of_smog_ds}. '
                  f'Dataset is too small to proceed. Try again later.')

            return exit(0)

        else:
            print(f'Length of currently available dataset is {size_of_smog_ds}. '
                  f'Dataset is fine. Proceeding.')

            return smog_ds


class DataPreprocessor:

    smog_ds = DataPackage.get_smog_data()

    def get_locations(self):

        locations = [{'longitude': float(item['school']['longitude']), 'latitude': float(item['school']['latitude'])}
                     for item in self.smog_ds['smog_data'] if 14.07 <= float(item['school']['longitude']) <= 24.09
                     and 49.0 <= float(item['school']['latitude']) <= 55.0]

        return locations

    def create_loc_data_frame(self):

        locations = self.get_locations()
        loc_df = pd.DataFrame(locations)
        loc_df['longitude'] = pd.to_numeric(loc_df['longitude'], errors='coerce')
        loc_df['latitude'] = pd.to_numeric(loc_df['latitude'], errors='coerce')
        loc_df.dropna(subset=['longitude', 'latitude'], inplace=True)
        loc_df = loc_df[(loc_df['longitude'] != 0) & (loc_df['latitude'] != 0)]

        return loc_df

    def transform_to_coordinates(self):

        loc_df = self.create_loc_data_frame()
        coordinates = loc_df.loc[:, ['longitude', 'latitude']]

        return coordinates

    def create_all_data_frame(self):

        all_df = pd.json_normalize(self.smog_ds['smog_data'])
        all_df['school.longitude'] = all_df['school.longitude'].astype(float)
        all_df['school.latitude'] = all_df['school.latitude'].astype(float)
        all_df.dropna(subset=['school.longitude', 'school.latitude'], inplace=True)
        all_df = all_df[(all_df['school.longitude'] != 0) & (all_df['school.latitude'] != 0)]

        return all_df

    def merge_data_sets(self):

        coordinates_labeled = self.create_loc_data_frame()
        all_data = self.create_all_data_frame()
        clean_all_data = pd.merge(coordinates_labeled, all_data,
                                  how='inner',
                                  left_on=['longitude', 'latitude'],
                                  right_on=['school.longitude', 'school.latitude'],
                                  suffixes=('_left', '_right'))
        clean_all_data = clean_all_data.drop_duplicates()

        return clean_all_data
