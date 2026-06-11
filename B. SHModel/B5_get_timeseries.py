# -*- coding: utf-8 -*-
"""
B5_get_timeseries.py

Builds the daily timeseries dataset used for model training and testing
(B6_get_actualUAs.py).

For each year of data provided, the following processing is applied:
    - All months are included (the heating-season filter is applied later
      during disaggregation, not here).
    - Hourly space heating (SH) and weather data are aggregated to daily
      averages (mean), to reduce the influence of time constants and daily
      variations. Mean (rather than median) is used so that high-consumption
      hours are appropriately weighted, which improves model accuracy.
    - Outliers (daily SH values above the 99th percentile per meter) are removed.

If two years of data are provided, they are concatenated into a single
timeseries before outlier removal and model training.
"""

import pandas as pd
import pickle

from B4_get_addProcessHM import localize_to_copenhagen


def wrangling(energy_data: pd.DataFrame, weather_data: pd.DataFrame,
              keep_months: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> pd.DataFrame:
    """
    Preprocess energy and weather data:
    - Keep only the specified months
    - Calculate daily averages for SH (space heating) and weather variables
    - Merge energy with weather

    energy_data: pd.DataFrame
        Must contain identifier columns, 'timestamp', and 'SH'.
    weather_data: pd.DataFrame
        Must contain at least 'timestamp' and weather variables.
    keep_months: list of int, optional
        Months to keep (default: all months, 1-12).
    on_cols: list of str, optional
        Identifier columns to retain and group by.

    Returns
    -------
    pd.DataFrame
        Daily timeseries indexed by date with SH and weather variables.
    """

    # Copy and enforce string type for identifier columns
    energy_df = energy_data.copy().reset_index(drop=True)
    for col in on_cols:
        if col in energy_df.columns:
            energy_df[col] = energy_df[col].astype(str)

    weather_df = weather_data.copy().reset_index(drop=True)

    # Keep only required columns (ids + timestamp + SH)
    energy_df = energy_df[on_cols + ['timestamp', 'SH']]

    # Set timezone to Copenhagen
    energy_df = localize_to_copenhagen(energy_df, "timestamp")
    weather_df = localize_to_copenhagen(weather_df, "timestamp")

    # Keep only the specified months
    energy_df = energy_df[energy_df['timestamp'].dt.month.isin(keep_months)]
    weather_df = weather_df[weather_df['timestamp'].dt.month.isin(keep_months)]

    # Add date column
    energy_df['date'] = energy_df['timestamp'].dt.date
    energy_df.drop(columns='timestamp', inplace=True)

    weather_df = pd.melt(weather_df, id_vars=['timestamp'], var_name='variable', value_name='value')
    weather_df['date'] = weather_df['timestamp'].dt.date
    weather_df.drop(columns='timestamp', inplace=True)

    # Daily averages
    energy_df = (
        energy_df
        .groupby(on_cols + ['date'])
        .agg({'SH': 'mean'})
        .reset_index()
        .set_index('date')
    )

    weather_df = (
        weather_df
        .groupby(['variable', 'date'])
        .agg({'value': 'mean'})
        .reset_index()
        .set_index('date')
        .pivot(columns='variable', values='value')
    )

    # Merge weather and SH data
    timeseries = energy_df.join(weather_df)

    return timeseries


def remove_faulty_data(data: pd.DataFrame, valueToFilter: str = 'SH',
                        on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> pd.DataFrame:
    """
    Remove outliers above the 99th percentile for each group of identifiers.

    data: pd.DataFrame
        Input dataframe containing identifier columns, a timestamp/date index,
        and the value column.
    valueToFilter: str, optional
        Column with numeric values to filter (default: 'SH').
    on_cols: list of str, optional
        Identifier columns to group by (default includes address, location_id,
        serial, BFEnummer).

    Returns
    -------
    pd.DataFrame
        Dataframe with outliers removed.
    """
    df = data.copy()

    # Ensure identifiers are strings (consistent merging across pipeline)
    for col in on_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Calculate 99th percentile per group
    df['q_upper'] = df.groupby(on_cols)[valueToFilter].transform(lambda x: x.quantile(0.99))

    # Calculate ratio to 99th percentile
    df['fct'] = df[valueToFilter] / df['q_upper']

    # Keep only values below or equal to threshold
    df = df[df['fct'] <= 1]

    # Drop helper columns
    df.drop(columns=['q_upper', 'fct'], inplace=True)

    return df


def get_timeseries(dataYR1: pd.DataFrame, weatherY1: pd.DataFrame, path_save_data: str,
                    dataYR2: pd.DataFrame = None, weatherY2: pd.DataFrame = None) -> None:
    """
    Build the daily timeseries dataset used for model training/testing
    (B6_get_actualUAs.py), and save it.

    dataYR1: pd.DataFrame
        Disaggregated SH data for year 1 (output of B4_get_addProcessHM.py /
        B2_get_disaggregated.py). Must contain "serial", "timestamp" and "SH".
    weatherY1: pd.DataFrame
        Hourly weather data for year 1 (output of B3_get_weather_data.py).
    path_save_data: str
        File path (.pkl) where the resulting daily timeseries is saved.
    dataYR2: pd.DataFrame, optional
        Disaggregated SH data for year 2. If None (default), only year 1 is used.
    weatherY2: pd.DataFrame, optional
        Hourly weather data for year 2. Required if `dataYR2` is provided.

    Returns
    -------
    None
        The resulting daily timeseries is written to `path_save_data` as a
        pickle file.
    """

    # Apply data wrangling: calculate daily averages and merge with weather
    ts_YR1 = wrangling(dataYR1, weatherY1, keep_months=list(range(1, 13)), on_cols=['serial'])

    if dataYR2 is not None:
        ts_YR2 = wrangling(dataYR2, weatherY2, keep_months=list(range(1, 13)), on_cols=['serial'])

        # Merge the two years of data
        timeseries = pd.concat([ts_YR1, ts_YR2])
    else:
        timeseries = ts_YR1

    # Remove faulty data (extreme outliers)
    timeseries = remove_faulty_data(timeseries, valueToFilter='SH', on_cols=['serial'])

    if 'Wind direction' in timeseries.columns:
        timeseries.drop(columns='Wind direction', inplace=True)

    with open(path_save_data, 'wb') as f:
        pickle.dump(timeseries, f)
