# -*- coding: utf-8 -*-
"""
B2_get_disaggregated.py

Disaggregates total hourly heat consumption into space heating (SH) and
domestic hot water (DHW) using the "summer method".

Summer method assumptions:
    - DHW consumption is roughly constant (insignificant variation) throughout
      the year.
    - SH is approximately 0 during the warmest month (August).

Steps:
    1) August is used as the reference month where SH ~ 0. If August data
       is unavailable for a meter, July is used as fallback, then June.
       If none of these months are available, the meter is skipped.
    2) For each day type (Mon-Thu, Fri, Sat, Sun), the median hourly heat
       consumption in August is taken as the DHW profile for that hour/day type.
    3) This DHW profile is applied to the full year; SH is calculated as the
       total heat consumption minus the DHW estimate.
    4) Negative SH values are clipped to 0.

Note: this assumes SH is negligible in August; users working in climates or
buildings where this assumption does not hold should adjust the reference
month accordingly.
"""

import pandas as pd


def assign_days(data):
    """Assigns day type based on weekday mapping. Uses category dtype for efficiency.
        
        data must have timestamp as index
    """
    
    day_mapping = {0: 'Weekday', 1: 'Weekday', 2: 'Weekday', 3: 'Weekday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    data['Day Type'] = data.index.weekday.map(day_mapping).astype('category')
    return data

def dayType(data, month):
    """Extracts data for a given month and assigns day type categories."""
    consumption_month = data.loc[data['timestamp'].dt.month == month, ['timestamp', 'heat_power']]
    consumption_month.set_index('timestamp', inplace=True)
    consumption_month = assign_days(consumption_month)
    return consumption_month

def summer_method(data: pd.DataFrame, consumption_month: pd.DataFrame, modified: bool = False) -> pd.DataFrame:
    """
    Apply the summer method to disaggregate DHW and SH usage for a single meter.

    data: pd.DataFrame
        Full-year meter data with "timestamp" and "heat_power" columns.
    consumption_month: pd.DataFrame
        Reference month data (typically August), indexed by timestamp, with
        "heat_power" and "Day Type" columns (output of dayType()).
    modified: bool, optional
        If True (default usage in apply_disagg), the median hourly
        consumption per day type is used directly as the DHW estimate
        (no normalization to a 0-100 profile). If False, a normalized 0-100
        profile is computed and rescaled to the median daily total.

    Returns
    -------
    pd.DataFrame
        `data` with added "DHW" and "SH" columns (both in the same units as
        "heat_power"). SH is clipped at a minimum of 0.

    Note: median (rather than mean) is used to compute the per-hour profile,
    as it is more robust to outliers.
    """
    DHWprofile_month = pd.DataFrame()
    
    for day in ['Weekday', 'Friday', 'Saturday', 'Sunday']:
        consumption_day = consumption_month.loc[consumption_month['Day Type'] == day, ['heat_power']]
        DHWprofile_day = consumption_day.groupby(consumption_day.index.hour).median()
        
        if modified:
            DHWprofile_day[day] = DHWprofile_day['heat_power']
        else:
            DHWprofile_day[day] = (DHWprofile_day['heat_power'] / DHWprofile_day['heat_power'].sum()) * 100

        DHWprofile_day.drop(columns='heat_power', inplace=True)
        DHWprofile_month = pd.concat([DHWprofile_month, DHWprofile_day], axis=1)
    
    # Convert to long format for merging
    DHWprofile_month_365 = DHWprofile_month.reset_index().melt(
        id_vars='timestamp', var_name='Day Type', value_name='DHW'
    )
    DHWprofile_month_365.rename(columns={'timestamp': 'hour'}, inplace=True)
    
    # Assign day types once before merging
    data.set_index('timestamp', inplace = True) 
    data = assign_days(data)
    data.reset_index(inplace = True)
    data['hour'] = data['timestamp'].dt.hour

    # Merge in a single step
    data = data.merge(DHWprofile_month_365, on=['Day Type', 'hour'], how='left')
    data.drop(columns=['Day Type', 'hour'], inplace=True)

    if not modified:
        consumption_month_daily_mean = consumption_month.resample('D').sum()['heat_power'].median()
        data['DHW'] = (data['DHW'] / 100) * consumption_month_daily_mean  # Normalize & scale

    # Calculate SH and ensure non-negative values
    data['SH'] = (data['heat_power'] - data['DHW']).clip(lower=0)

    return data

def apply_disagg(file: str, file_save: str, on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> None:
    """
    Apply the SH/DHW disaggregation (summer method) to an entire dataset, one
    meter at a time, and save the result.

    file: str
        File path (.parquet) of the processed hourly meter data (output of
        B1_get_processedHM.py). Must contain "timestamp", "heat_power",
        "serial" and the columns listed in `on_cols`.
    file_save: str
        File path (.parquet) where the data, enriched with "DHW" and "SH"
        columns, is saved.
    on_cols: list of str, optional
        Identifier columns to retain alongside "timestamp" and "heat_power"
        (default: ['address', 'location_id', 'serial', 'BFEnummer']).

    Returns
    -------
    None
        The enriched dataset is written to `file_save` as a parquet file.
    """
    energy_data = pd.read_parquet(file, engine='pyarrow')
    
    # Optimize serial number retrieval
    serialnrs = energy_data['serial'].unique()
    
    # Store results in a list instead of merging in every iteration
    results = []
    
    for i, serialnr in enumerate(serialnrs, 1):
        subset_energy_data = energy_data.loc[energy_data['serial'] == serialnr, 
                                             ['timestamp', 'heat_power'] + on_cols]
        subset_energy_data = subset_energy_data.reset_index(drop=True)

        # Try August first, fall back to July, then June
        reference_month = None
        for month in [8, 7, 6]:
            candidate = dayType(subset_energy_data, month=month)
            if not candidate.empty:
                reference_month = candidate
                break

        if reference_month is None:
            print(f"Skipping meter {serialnr} - no data in June, July or August")
            continue

        # Apply Summer Method
        processed_data = summer_method(subset_energy_data.copy(), reference_month, modified=True)
        results.append(processed_data)

        print(f"Processed {i}/{len(serialnrs)} meters")
    
    # Concatenate all processed data once at the end
    propty_SH = pd.concat(results, ignore_index=True)
    
    # Align serial dtype before merging
    propty_SH['serial'] = propty_SH['serial'].astype(energy_data['serial'].dtype)
        
    # Merge with original data in a single operation
    energy_wSH_data = energy_data.merge(
        propty_SH[['timestamp'] + on_cols + ['DHW', 'SH']], 
        on=['timestamp'] + on_cols, 
        how='left'
    )

    energy_wSH_data.to_parquet(file_save)


