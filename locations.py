from geopy import Nominatim


class VoivodLoc:

    def __init__(self):

        self.loc = Nominatim(user_agent="GetLoc")

    def locations(self):

        voi_capitals = ['Wrocław', 'Opole', 'Katowice', 'Kielce', 'Kraków', 'Lublin', 'Poznań', 'Łódź', 'Warszawa',
                        'Rzeszów', 'Białystok', 'Gorzów Wielkopolski', 'Bydgoszcz', 'Szczecin', 'Olsztyn', 'Gdańsk']

        capital_loc = {}

        for capital_city in sorted(voi_capitals):

            get_loc = self.loc.geocode(capital_city)
            capital_loc[capital_city] = {'longitude': get_loc.longitude, 'latitude': get_loc.latitude}

        return capital_loc


class UserLoc(VoivodLoc):

    def __init__(self):
        super().__init__()


    def get_user_loc(self):

        get_loc = self.loc.geocode(input('Enter your location: '))
        user_loc = {'longitude': get_loc.longitude, 'latitude': get_loc.latitude}
        return user_loc