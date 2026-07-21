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

A free DMI Open Data API key can be obtained at:
https://opendatadocs.dmi.govcloud.dk/en/Authentication
"""

import pandas as pd
import requests


def dmi_data(stationID: str, start_time: str, end_time: str, parameter: str, api_key: str) -> pd.DataFrame:
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
        https://opendatadocs.dmi.govcloud.dk/en/Data/Meteorological_Observation_Data
        for the full list of available parameters.
    api_key: str
        DMI Open Data API key.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "timestamp" and "properties.value", containing
        the raw (sub-hourly) observations for the requested parameter.
    """
    URL = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items'

    body = {'datetime': f'{start_time}/{end_time}',
            'limit': '100000',
            'stationId': stationID,
            'parameterId': parameter,
            'api-key': api_key,
            }

    response = requests.get(URL, params=body)
    weather_data = pd.json_normalize(response.json()['features'])

    # MAKE "PROPERTIES OBSERVED" INDEX
    weather_data.index = weather_data['properties.observed']

    # DELETE UNUSED COLUMNS
    weather_data = weather_data.drop(['id', 'type', 'geometry.coordinates', 'geometry.type',
                      'properties.created', 'properties.observed', 'properties.parameterId', 'properties.stationId'],
                     axis='columns')

    # TRANSFORM INDEX TO DATETIME FORMAT
    weather_data.index.names = ['timestamp']
    weather_data.index = pd.to_datetime(weather_data.index)
    weather_data = weather_data.reset_index()

    return weather_data


def get_weather(path_weather_save: str, year: int, api_key: str,
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
    api_key: str
        DMI Open Data API key.
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

    # GET DATA FROM DMI, parameters https://opendatadocs.dmi.govcloud.dk/en/Data/Meteorological_Observation_Data
    temp = dmi_data(station_temperature, start_time, end_time, 'temp_dry', api_key)
    wind_dir = dmi_data(station_wind_direction, start_time, end_time, 'wind_dir', api_key)
    wind_speed = dmi_data(station_wind_speed, start_time, end_time, 'wind_speed', api_key)
    sun = dmi_data(station_solar_radiation, start_time, end_time, 'radia_glob', api_key)

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

    # MERGE WEATHER DATA
    weather_hourly = pd.concat([temp_hourly, wind_dir_hourly, wind_speed_hourly, sun_hourly], axis=1)
    weather_hourly = weather_hourly.reset_index()
    weather_hourly = weather_hourly.ffill()

    # CONVERT TO LOCAL TIME ZONE AND FILL NAN VALUES
    weather_hourly['timestamp'] = pd.to_datetime(weather_hourly['timestamp']).dt.tz_convert('Europe/Copenhagen')

    # REMOVE TIMEZONE INFORMATION
    weather_hourly['timestamp'] = weather_hourly['timestamp'].dt.tz_localize(None)

    # CHECK IF THERE ARE DUPLICATED TIME ENTRIES
    # print(weather_hourly[weather_hourly.duplicated(subset='timestamp',keep=False)])

    # REMOVE DUPLICATED TIME ENTRIES BY AVERAGING THE VALUES OF THE REMAINING COLUMNS
    weather_cleaned = weather_hourly.groupby('timestamp').mean().reset_index()

    weather_cleaned.to_parquet(path_weather_save)
