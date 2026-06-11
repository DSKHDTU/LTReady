# -*- coding: utf-8 -*-
"""
B1_get_processedHM.py

Performs data quality checks, processing and wrangling of raw hourly heat
meter (HM) data for a single year.

Data cleaning steps applied:
    - Remove redundant columns and exact-duplicate rows.
    - Add a complete hourly timestamp index per meter (NaN for missing hours).
    - Remove erroneous measurements (extreme outliers, return temperature
      higher than inlet temperature, etc.).
    - Remove meters with more than ~6 months (50%) of consecutive missing data.
    - Remove meters with zero total heat_power for the entire period.
"""

import pandas as pd
import pickle
import numpy as np

REQUIRED_METER_COLUMNS = [
    "timestamp",
    "energy",
    "volume",
    "heat_power",
    "volume_flow",
    "inlet_flow_temperature",
    "return_flow_temperature",
]


def has_large_missing_streak(sub_df):
    sub_df = sub_df.sort_values('timestamp')  # Ensure order
    sub_df['missing'] = sub_df['heat_power'].isna()
    
    # Identify groups of consecutive missing values
    sub_df['group'] = (sub_df['missing'] != sub_df['missing'].shift()).cumsum()
    
    # Find longest missing streak
    missing_streaks = sub_df[sub_df['missing']].groupby('group').size()
    
    # Check if any missing streak is more than 50% of total rows
    return missing_streaks.max() > (0.5 * len(sub_df)) if not missing_streaks.empty else False


def drop_redundant(data, drop_columns):
    
    # Remove redundant data
    if drop_columns  != None: 
        data_raw = data.drop(columns = drop_columns)
    else:
        data_raw = data
    
    # Remove duplicates
    data_raw = data_raw.drop_duplicates().reset_index(drop=True)
    
    return data_raw

def time_sync(data_raw, grouping_columns, year):
    # Create the full timestamp sequence
    full_time_range = pd.date_range(start=f"{year}-01-01 01:00:00", end=f"{int(year)+1}-01-01 00:00:00", freq="h")
     
    # Get unique location details
    unique_locations = data_raw[grouping_columns].drop_duplicates()
     
    # Create a DataFrame with all possible timestamp-location combinations
    full_time_df = unique_locations.merge(pd.DataFrame({'timestamp': full_time_range}), how='cross')
     
    # Merge with the original dataset to retain existing values
    data_raw = full_time_df.merge(data_raw, on=grouping_columns + ['timestamp'], how='left')
    
    return data_raw


def remove_err_data(data_raw):
    for column in ['heat_power', 'volume_flow']:    
         q_up = data_raw.groupby('serial')[column].transform(lambda x: x.quantile(0.99))  # Use temp variable
         data_raw[column] = data_raw[column].where(data_raw[column].between(0, q_up), np.nan)  # Filter outliers
     
    data_raw['inlet_flow_temperature'] = data_raw['inlet_flow_temperature'].where(
         data_raw['inlet_flow_temperature'].between(0, 150, inclusive='neither'), np.nan
    )  # Forventer ikke temperatur over 150 C men vi har sat det lidt højere just in case
     
    data_raw['return_flow_temperature'] = data_raw['return_flow_temperature'].where(
         data_raw['return_flow_temperature'] < data_raw['inlet_flow_temperature'], np.nan
    )  # Keep values as-is if return_flow_temperature < inlet_flow_temperature                   
    
    return data_raw

def remove_consc_missing(data_raw):
    # Find serials to remove
    serials_to_remove = data_raw.groupby('serial').filter(has_large_missing_streak)['serial'].unique()
     
    # Remove them from the dataset
    data_raw = data_raw[~data_raw['serial'].isin(serials_to_remove)]
    
    return data_raw

def remove_faulty_power(data_raw,grouping_columns):
    meters_to_remove = data_raw.groupby(grouping_columns)['heat_power'].sum().reset_index()
    meters_to_remove = meters_to_remove[meters_to_remove['heat_power'] == 0]['serial']
     
    # Remove those meters from the data_raw
    data_raw = data_raw[~data_raw['serial'].isin(meters_to_remove)]
    
    return data_raw

def get_merged_ID(data_raw: pd.DataFrame, stamdata: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the processed meter data with the building master ID table on
    "address", to attach the "BFEnummer" identifier.

    data_raw: pd.DataFrame
        Processed meter data. If it contains an "address" column, it is
        merged with `stamdata`; otherwise returned unchanged.
    stamdata: pd.DataFrame
        Master ID dataframe containing "address" and "BFEnummer" columns.

    Returns
    -------
    pd.DataFrame
        `data_raw` with an added "BFEnummer" column, if "address" was present.
    """
    if 'address' in data_raw.columns:  # Ensure 'address' column exists
         data_raw = pd.merge(data_raw, stamdata[['address', 'BFEnummer']], on='address', how='left')
         
    return data_raw
    
def proc_raw_meter(path_meter: str, path_save_meter: str,
                    grouping_columns: list, year: int, stamdata: pd.DataFrame = None) -> None:
    """
    Load raw hourly heat meter data for a single year, clean it, and save the
    processed result.

    path_meter: str
        File path (.pkl/.parquet) of the raw hourly meter data for `year`.
        Must contain at least the columns "timestamp", "serial", "heat_power",
        "volume_flow", "inlet_flow_temperature" and "return_flow_temperature",
        plus the columns listed in `grouping_columns`.
    path_save_meter: str
        File path (.parquet) where the processed meter data is saved.
    grouping_columns: list of str
        Columns that uniquely identify a meter (e.g. ["serial"]).
    year: int
        The calendar year covered by the meter data, used to build the
        complete hourly timestamp index.
    stamdata: pd.DataFrame, optional
        Master ID dataframe with "address" and "BFEnummer" columns. If
        provided, the processed data is merged with it to attach the
        "BFEnummer" identifier. Default None (no merge performed).

    Returns
    -------
    None
        The processed meter data is written to `path_save_meter` as a
        parquet file.
    """

    data = pd.read_parquet(path_meter, engine='pyarrow')

    # 1) Drop any columns that are not required for processing, and remove duplicates
    keep_columns = REQUIRED_METER_COLUMNS + grouping_columns
    drop_columns = [col for col in data.columns if col not in keep_columns]
    data_raw = drop_redundant(data, drop_columns)

    # 2) Add full time period
    data_raw = time_sync(data_raw, grouping_columns, year)

    # 3) Remove erroneous data
    data_raw = remove_err_data(data_raw)

    # 4) Remove meters with more than 6 months of missing data or large gaps
    data_raw = remove_consc_missing(data_raw)

    # 5) Remove meters with a sum of heat_power equal to 0
    data_raw = remove_faulty_power(data_raw, grouping_columns)

    if stamdata is not None:
        # 6) Merge processed data with relevant building information and BFE number
        data_raw = get_merged_ID(data_raw, stamdata)

    data_raw.to_parquet(path_save_meter)
 



