# -*- coding: utf-8 -*-
"""
B3_get_weather_data.py

Retrieves hourly weather observations (outdoor temperature, wind direction,
wind speed and solar radiation) from the DMI (Danish Meteorological Institute)
Open Data API for a given year, and saves a cleaned, hourly weather dataset.

Each weather parameter can be retrieved from a different DMI weather station,
since not all stations measure all parameters. DMI station IDs can be found at:
https://www.dmi.dk/friedata/observationer/ or in the DMI station catalogue
(https://www.dmi.dk/fileadmin/Rapporter/TR/tr13-13.pdf).

API update
----------
DMI has moved its Open Data API to a new hostname, opendataapi.dmi.dk, which
no longer requires an API key. The old hostname (dmigw.govcloud.dk) is being
phased out. Documentation:
https://www.dmi.dk/friedata/dokumentation/meteorological-observation-api
Fair-use limit: 500 requests per 5 seconds.
"""

import pandas as pd
import requests

BASE_URL = 'https://opendataapi.dmi.dk/v2/metObs/collections/observation/items'


def dmi_data(stationID: str, start_time: str, end_time: str, parameter: str) -> pd.DataFrame:
    """
    Retrieve a single weather parameter for a single station from the DMI
    Open Data API.

    stationID: str
        DMI weather station ID (e.g. "06180").
    start_time: str
        Start of the observation period, in ISO 8601 format with UTC
        timezone (e.g. "2022-01-01T00:00:00Z").
    end_time: str
        End of the observation period, in ISO 8601 format with UTC timezone.
    parameter: str
        DMI parameter ID to retrieve (e.g. "temp_dry", "wind_dir",
        "wind_speed", "radia_glob"). See
        https://www.dmi.dk/friedata/dokumentation/meteorological-observation-api
        for the full list of available parameters.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "timestamp" and "properties.value", containing
        the raw (sub-hourly) observations for the requested parameter.
    """
    params = {'datetime': f'{start_time}/{end_time}',
              'limit': 300000,
              'stationId': stationID,
              'parameterId': parameter,
              }

    # FETCH ALL PAGES (follow the "next" link if the result is paginated)
    features = []
    url = BASE_URL
    while url:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        features.extend(data.get('features', []))

        next_links = [link['href'] for link in data.get('links', []) if link.get('rel') == 'next']
        url = next_links[0] if next_links else None
        params = None  # the "next" link already contains all query parameters

    if not features:
        raise ValueError(f'No data returned from DMI for station {stationID}, '
                         f'parameter {parameter}, period {start_time} to {end_time}.')

    weather_data = pd.json_normalize(features)

    # KEEP ONLY TIMESTAMP AND VALUE
    weather_data = weather_data[['properties.observed', 'properties.value']]
    weather_data = weather_data.rename(columns={'properties.observed': 'timestamp'})

    # TRANSFORM TIMESTAMP TO DATETIME FORMAT AND SORT
    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
    weather_data = weather_data.sort_values('timestamp').reset_index(drop=True)

    return weather_data


def get_weather(path_weather_save: str, year: int,
                station_temperature: str, station_wind_direction: str,
                station_wind_speed: str, station_solar_radiation: str) -> None:
    """
    Retrieve hourly outdoor temperature, wind direction, wind speed and solar
    radiation for a given year from DMI Open Data, and save the merged,
    cleaned hourly weather dataset.

    path_weather_save: str
        File path (.parquet) where the resulting hourly weather dataset is saved.
    year: int
        Calendar year to retrieve weather data for.
    station_temperature: str
        DMI station ID used to retrieve outdoor temperature ("temp_dry").
    station_wind_direction: str
        DMI station ID used to retrieve wind direction ("wind_dir").
    station_wind_speed: str
        DMI station ID used to retrieve wind speed ("wind_speed").
    station_solar_radiation: str
        DMI station ID used to retrieve global solar radiation ("radia_glob").

    Note
    ----
    The four station IDs may all be the same, or different, depending on
    which stations near the area of interest measure which parameters.

    Returns
    -------
    None
        The hourly weather dataset is written to `path_weather_save` as a
        parquet file, with columns "timestamp", "Temperature",
        "Wind direction", "Wind speed" and "Sun".
    """

    # SET START AND END TIME (TIMEZONE = GMT)
    start_time = str(year-1) + '-12-31T23:00:00Z'
    end_time = str(year+1) + '-01-01T01:00:00Z'

    # GET DATA FROM DMI, parameters: https://www.dmi.dk/friedata/dokumentation/meteorological-observation-api
    temp = dmi_data(station_temperature, start_time, end_time, 'temp_dry')
    wind_dir = dmi_data(station_wind_direction, start_time, end_time, 'wind_dir')
    wind_speed = dmi_data(station_wind_speed, start_time, end_time, 'wind_speed')
    sun = dmi_data(station_solar_radiation, start_time, end_time, 'radia_glob')

    # GIVE COLUMNS NAMES
    temp.rename(columns={'properties.value': 'Temperature'}, inplace=True)
    wind_dir.rename(columns={'properties.value': 'Wind direction'}, inplace=True)
    wind_speed.rename(columns={'properties.value': 'Wind speed'}, inplace=True)
    sun.rename(columns={'properties.value': 'Sun'}, inplace=True)

    # HOURLY VALUES
    temp_hourly = temp.resample('h', on='timestamp').mean()
    wind_dir_hourly = wind_dir.resample('h', on='timestamp').mean()
    wind_speed_hourly = wind_speed.resample('h', on='timestamp').mean()
    sun_hourly = sun.resample('h', on='timestamp').mean()

    # MERGE WEATHER DATA AND FILL NAN VALUES
    weather_hourly = pd.concat([temp_hourly, wind_dir_hourly, wind_speed_hourly, sun_hourly], axis=1)
    weather_hourly = weather_hourly.reset_index()
    weather_hourly = weather_hourly.ffill()

    # CONVERT TO LOCAL TIME ZONE
    weather_hourly['timestamp'] = pd.to_datetime(weather_hourly['timestamp']).dt.tz_convert('Europe/Copenhagen')

    # REMOVE TIMEZONE INFORMATION
    weather_hourly['timestamp'] = weather_hourly['timestamp'].dt.tz_localize(None)

    # REMOVE DUPLICATED TIME ENTRIES (DST change in autumn) BY AVERAGING
    weather_cleaned = weather_hourly.groupby('timestamp').mean().reset_index()

    weather_cleaned.to_parquet(path_weather_save)