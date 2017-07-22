""" Get geographical coordinates of a city using Google Maps API. """

import pandas as pd
import numpy as np
from geopy import geocoders
from geopy import exc
import time

# INSERT GOOGLE MAPS API CODE HERE:
API_CODE = ''


def get_city_geoloc(file_in, file_out):
    g = geocoders.GoogleV3(API_CODE)
    cities = pd.read_csv(file_in)

    latitude_city, longitude_city = [], []

    f = open(file_out, 'w')

    for _, c in cities.iterrows():

        city = c.City
        country_code = c.CountryCode

        if pd.isnull(city):
            city = ''
        if pd.isnull(country_code):
            country_code = ''

        if pd.isnull(city) and pd.isnull(country_code):
            l1, l2 = np.nan, np.nan
            latitude_city.append(l1)
            longitude_city.append(l2)
        else:
            for _ in range(5):
                try:
                    coord = g.geocode(city, components={'country': country_code})
                    break
                except exc.GeocoderTimedOut:
                    time.sleep(1)
                    pass

            try:
                l1, l2 = coord.latitude, coord.longitude
                latitude_city.append(l1)
                longitude_city.append(l2)
            except AttributeError:
                l1, l2 = np.nan, np.nan
                latitude_city.append(l1)
                longitude_city.append(l2)

        print(l1, l2)
        f.write(str(l1) + ',' + str(l2) + '\n')

    f.close()
