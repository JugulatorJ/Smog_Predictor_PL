import json
import os
import pandas as pd
import requests
from dotenv import load_dotenv


pd.set_option('display.max_columns', None)
load_dotenv()


class Credentials:

    @staticmethod
    def take_key(name):

        key = os.getenv(name)

        return key


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

    @staticmethod
    def get_additional_weather_data(user_location):
        key = Credentials.take_key('WEATHER_API_KEY')
        lat = user_location['latitude']
        lon = user_location['longitude']
        additional_weather_data = json.loads(requests.get(f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}'
                                                          f'&lon={lon}&exclude=minutely,hourly,daily&units=metric'
                                                          f'&appid={key}').text)

        return additional_weather_data


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

        clean_all_data.drop(columns=['school.name', 'school.street', 'school.post_code', 'school.city',
                            'school.longitude', 'school.latitude'], inplace=True)
        clean_all_data['timestamp'] = pd.to_datetime(clean_all_data['timestamp'])
        clean_all_data.drop_duplicates(inplace=True)
        clean_all_data.reset_index(drop=True, inplace=True)

        return clean_all_data

    @staticmethod
    def create_weather_df(user_location):

        weather_ds = DataPackage.get_additional_weather_data(user_location)
        print(user_location)
        weather_df = pd.DataFrame(weather_ds)
        weather_df = weather_df.dropna()
        weather_df = weather_df.drop(['lat', 'lon', 'timezone', 'timezone_offset'], axis=1)
        weather_df = weather_df.drop(['clouds', 'dew_point', 'dt', 'feels_like', 'sunrise', 'sunset', 'uvi', 'visibility',
                                      'weather', 'wind_deg', 'wind_gust', 'wind_speed'], axis=0)
        weather_df = weather_df.transpose()
        weather_df = weather_df.rename(columns={"humidity": "data.humidity_avg", "pressure": "data.pressure_avg",
                                                "temp": "data.temperature_avg"})

        return weather_df
