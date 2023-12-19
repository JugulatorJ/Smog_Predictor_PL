import json
import pandas as pd
import requests


class DataPackage:

    def __init__(self):

        pass

    @staticmethod
    def get_smog_data():

        smog_ds = json.loads(requests.get('https://public-esa.ose.gov.pl/api/v1/smog').text)
        if len(smog_ds['smog_data']) <=1200:
            print('No data available')
            return exit(1)
        else:
            return smog_ds


class DataPreprocessor(DataPackage):

    def __init__(self):

        super().__init__()


    def get_locations(self):

        smog_ds = self.get_smog_data()
        locations = [{'longitude': float(item['school']['longitude']), 'latitude': float(item['school']['latitude'])}
                     for item in smog_ds['smog_data'] if 14.07 <= float(item['school']['longitude']) <= 24.09
                     and 49.0 <= float(item['school']['latitude']) <= 55.0]
        return locations

    def create_data_frame(self):

        locations = self.get_locations()

        df = pd.DataFrame(locations)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df.dropna(subset=['longitude', 'latitude'], inplace=True)
        df = df[(df['longitude'] != 0) & (df['latitude'] != 0)]

        return df

    def transform_to_coordinates(self):

        df = self.create_data_frame()
        coordinates = df.loc[:, ['longitude', 'latitude']]

        return coordinates
