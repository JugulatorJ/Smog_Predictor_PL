from time import time
import json
import pandas as pd
import requests
from matplotlib import pyplot as plt


class DataPackage:

    def __init__(self):

        pass

    @staticmethod
    def get_smog_data():

        smog_ds = json.loads(requests.get('https://public-esa.ose.gov.pl/api/v1/smog').text)

        return smog_ds


class DataPreprocessor(DataPackage):

    def __init__(self):

        super().__init__()

    def get_locations(self):

        smog_ds = self.get_smog_data()

        locations = [{'longitude': item['school']['longitude'], 'latitude': item['school']['latitude']}
                     for item in smog_ds['smog_data']]

        return locations

    def create_data_frame(self):

        locations = self.get_locations()

        df = pd.DataFrame(locations)

        # Convert longitude and latitude to numeric values
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df.dropna(subset=['longitude', 'latitude'], inplace=True)
        df = df[(df['longitude'] != 0) & (df['latitude'] != 0)]

        return df

    def transform_to_coordinates(self):

        df = self.create_data_frame()
        coordinates = df.loc[:, ['longitude', 'latitude']]

        return coordinates
