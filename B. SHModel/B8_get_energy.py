# -*- coding: utf-8 -*-
"""
B8_get_energy.py

Calculates the predicted annual space heating (SH) energy consumption for
each meter/building, using the trained model (B6_get_actualUAs.py) and actual
hourly weather data for the year of interest. The result is normalized to a
reference (long-term average) number of heating degree days, so that years
with different weather can be compared on a like-for-like basis.
"""

import pandas as pd
from B7_get_actualHeat import heatloss_calculation, get_modelUAs


def add_cumulative_energy(df: pd.DataFrame, on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> pd.DataFrame:
    """
    Calculate cumulative energy consumption [Wh] from hourly heat power [W],
    per group defined by `on_cols`.

    df: pd.DataFrame
        Hourly data with "timestamp" and "heat_power" columns, plus the
        identifier columns in `on_cols`.
    on_cols: list of str, optional
        Identifier columns to group by (default: ['address', 'location_id', 'serial', 'BFEnummer']).

    Returns
    -------
    pd.DataFrame
        `df` with an added "energy" column (cumulative sum of heat_power * dt
        in hours, in Wh). Time gaps greater than 1 hour are treated as missing
        (NaN) rather than integrated.
    """
    # Ensure sorting by on_cols and timestamp
    df = df.sort_values(on_cols + ['timestamp']).reset_index(drop=True)

    # Time difference in seconds, per group
    dt_s = df.groupby(on_cols)['timestamp'].diff().dt.total_seconds()
    dt_h = dt_s / 3600  # hours

    # Identify gaps: if dt > 1 hour, treat the corresponding energy as missing
    gap_mask = dt_s > 3600

    # Energy in Wh = Power [W] * dt [h], NaN where there is a time gap
    df['_energy_step'] = (df['heat_power'] * dt_h).where(~gap_mask)

    # Cumulative sum per group
    df['energy'] = df.groupby(on_cols)['_energy_step'].cumsum()
    df = df.drop(columns='_energy_step')

    return df

def annual_energy(data: pd.DataFrame, HD: float, on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> pd.DataFrame:
    """
    Calculate normalized annual energy consumption in MWh for each unique building/group.

    data: pd.DataFrame
        Time series data containing at least ['timestamp', 'energy'] and identifier columns.
    HD: float
        Observed heating degree days for the measurement period.
    on_cols: list of str, optional
        The identifier columns to group by (default includes address, location_id, serial, BFEnummer).

    Returns
    -------
    pd.DataFrame
        A dataframe with one row per group and normalized annual energy consumption
        (column "Energy_MWh"), plus the first and last observed timestamps.
    """

    # The Danish reference (normal-year) value of 2552 heating degree days,
    # as defined by the Danish Technological Institute for the 1991-2020 period.
    HDNormal = 2552

    results = []
    for keys, group in data.groupby(on_cols):
        group = group.dropna(subset=['energy', 'timestamp'])
        if group.empty:
            continue
        
        group = group.sort_values(by='timestamp')
        first_entry = group.iloc[0]
        last_entry = group.iloc[-1]
        
        annual_energy_consumption = last_entry['energy'] - first_entry['energy']
        
        # Ensure keys is a tuple if multiple grouping columns
        if isinstance(keys, tuple):
            row_data = dict(zip(on_cols, keys))
        else:
            row_data = {on_cols[0]: keys}
        
        row_data.update({
            'First Date': first_entry['timestamp'],
            'Last Date': last_entry['timestamp'],
            'Energy_MWh': annual_energy_consumption / 10**6 * HDNormal / HD
        })
        
        results.append(row_data)
    
    return pd.DataFrame(results)

def get_annual_energy(file_path_weather: str, file_path_model: str, HD: float, file_path_save: str,
                      weather_dependent_model: bool = False,
                      on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> None:
    """
    Calculate the predicted, normalized annual SH energy consumption for each
    meter/building, using actual hourly weather data, and save the result.

    file_path_weather: str
        File path (.parquet) of the hourly weather data for the year of
        interest (output of B3_get_weather_data.py).
    file_path_model: str
        File path (.pkl) of the trained model results (output of
        B6_get_actualUAs.py).
    HD: float
        Observed heating degree days for the year covered by
        `file_path_weather`.
    file_path_save: str
        File path (.parquet) where the resulting annual energy dataframe is saved.
    weather_dependent_model: bool, optional
        If True, exclude months not used during model training (months with
        no expected space heating demand). Default False.
    on_cols: list of str, optional
        Identifier columns (default: ['address', 'location_id', 'serial', 'BFEnummer']).

    Returns
    -------
    None
        The resulting dataframe (one row per identifier combination, with
        columns "First Date", "Last Date" and "Energy_MWh") is written to
        `file_path_save` as a parquet file.
    """
    
    # Get the weather file
    weather = pd.read_parquet(file_path_weather, engine='pyarrow')
    
    if weather_dependent_model:
        # Exclude the months which were not included in model training (no space heating)
        weather = weather[weather['timestamp'].dt.month.isin([1, 2, 3, 4, 9, 10, 11, 12])].reset_index(drop=True)
        
    # Extract the timestamp for later use
    time = weather['timestamp'].reset_index(drop=True)
    
    # Remove redundant columns
    weather.drop(columns=['timestamp','Wind direction'], inplace = True)
    
    # Rename the columns to match the model's naming of parameters
    weather.rename(columns = {'Temperature':'Tout','Wind speed':'Ws', 'Sun':'Ig'}, inplace = True)
    
    # Get the models and select only parameter for 'Coef' and calculate the heatloss including solar load
    df = heatloss_calculation(
        UAs=get_modelUAs(file_path_model, on_cols=on_cols).query('index == "Coef"'),
        conditions=weather,
        inclSun=True,
        on_cols=on_cols,
        onlySave=False,
    )
    
    # Get the annual energy
    energy = []
    for keys, group in df.groupby(on_cols):
        # Reset index
        group.reset_index(drop=True,inplace=True)
        
        # Add timestamp
        group['timestamp'] = time
        
        # Set heat loss to 0 if negative
        #df['TotalHeatloss'] = df['TotalHeatloss'].clip(lower=0)
        
        # Rename columns
        group.rename(columns = {'TotalHeatloss':'heat_power'}, inplace = True)
        
        # Calculate cumulative energy
        group = add_cumulative_energy(group, on_cols=on_cols)
        
        energy.append(group)
    
    energy = pd.concat(energy)   
    
    # Calculate the annual energy consumption and save the dataframe
    annual_energy(energy, HD, on_cols=on_cols).to_parquet(file_path_save)



    
