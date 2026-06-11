# -*- coding: utf-8 -*-
"""
C1_get_actualDHTemperatures.py

Calculates the minimum supply and return temperatures achievable for each
component, based on:
    - the predicted "actual" peak space heating heat loss
      (B7_get_actualHeat.py / A8_get_aggXML.py aggregated heat loss), and
    - the nominal radiator design parameters (K, design mass flow) derived
      from the EPC data (A5/A8).

Two methods are supported (see `method` argument of get_LTDH_temperatures()):
    - "mnorm": uses the EPC-derived nominal mass flow directly.
    - "deltaT": derives a design mass flow from a fixed temperature drop
      (20 degC for two-string/"Dual" heating systems, 10 degC for
      one-string/"Unified" systems), bounded by the EPC-derived nominal
      mass flow.

"""

import pickle
import pandas as pd
import numpy as np

from A5_get_DHTemperatures import get_supply_return_temperature, get_radiator_design

def itemToFloat(item, mapping: dict = None):
    """
    Convert "Type of system" labels (e.g. "Unified"/"Dual" or 1/2) to a list
    of floats, for downstream averaging.

    item: list, np.ndarray, None, or NaN
        The "Type of system" value(s) for a component. Supported inputs:
        a list containing 'Unified'/'Dual', 1/2, or '1'/'2', or None/NaN.
    mapping: dict, optional
        Mapping from string labels to numeric codes. Default
        {"Unified": 1.0, "Dual": 2.0}.

    Returns
    -------
    list of float, or np.nan
        The converted values, or np.nan if `item` is None/NaN or no valid
        values were found.
    """
    
    if mapping is None:
        mapping = {"Unified": 1.0, "Dual": 2.0}

    # Handle None / NaN
    if item is None or (isinstance(item, float) and np.isnan(item)):
        return np.nan
    elif (isinstance(item, np.ndarray)):
        # Convert to list
        item = list(item)
    elif not isinstance(item, list):
        raise ValueError(f"Expected list, got {type(item)}")

    values = []
    for x in item:
        if x in mapping:
            values.append(mapping[x])
        elif isinstance(x, (int, float)):
            values.append(float(x))
        elif isinstance(x, str) and x.strip().isdigit():
            values.append(float(x))
        # silently drop invalid values

    return values if values else np.nan


def get_LTDH_temperatures(path_load_actHL: str, path_load_min: str, path_load_max: str,
                           path_save_min: str, path_save_max: str, path_load_BBR: str = None,
                           col_HL: str = 'TotalHeatloss', method: str = 'mnorm',
                           paramb: list = None, Tinb: list = None, rad_nb: list = None) -> None:
    """
    Calculate the minimum supply and return temperatures for each component
    and save the result.

    path_load_actHL: str
        File path (.parquet) of the aggregated predicted heat loss (output of
        B9_get_aggHM.py, data_type="heatloss").
    path_load_min: str
        File path (.pkl) of the aggregated minimum original-design table
        (output of A8_get_aggXML.py, aggMaxMin(minOrMax="Min")).
    path_load_max: str
        File path (.pkl) of the aggregated maximum original-design table
        (output of A8_get_aggXML.py, aggMaxMin(minOrMax="Max")).
    path_save_min: str
        File path (.parquet) where the result based on the minimum
        original-design table is saved.
    path_save_max: str
        File path (.parquet) where the result based on the maximum
        original-design table is saved.
    path_load_BBR: str, optional
        File path (.pkl) of the aggregated BBR table (output of
        A8_get_aggXML.py, aggBBR()). Required if method="deltaT".
    col_HL: str, optional
        Name of the heat loss column to use from `path_load_actHL`
        (default "TotalHeatloss").
    method: str, optional
        Either "mnorm" (use the EPC-derived nominal mass flow directly) or
        "deltaT" (derive a design mass flow from a fixed temperature drop;
        see module docstring). Default "mnorm".
    paramb: list of str, optional
        If `Tinb` is provided, restrict `path_load_actHL` to rows where
        "param_stats" is in `paramb` (e.g. ['Coef']).
    Tinb: list of float, optional
        If provided, restrict `path_load_actHL` to rows where "Tin" is in
        `Tinb`.
    rad_nb: list of float, optional
        If provided, restrict the min/max design tables to rows where
        "rad_n" is in `rad_nb`.

    Returns
    -------
    None
        Writes two parquet files: `path_save_min` (based on the minimum
        original-design table) and `path_save_max` (based on the maximum
        original-design table), each with added "Actual Tsupply" and
        "Actual Treturn" columns (and, for method="deltaT", an "Actual
        Design massFlow" column).
    """

    # Import
    heatloss = pd.read_parquet(path_load_actHL, engine='pyarrow')
    
    if Tinb is not None:
        heatloss = heatloss.query('Tin==@Tinb & param_stats==@paramb')
    
    #Merge the EPC radiator design values and actual heatloss calculations
    if path_load_min is not None:
        with open(path_load_min, 'rb') as f:
            minDesign = pickle.load(f)
            
        # The nominal K value is based on the nominal radiator exponent which is 1.3
        # 1. Create a lookup table (Series)
        k_13_lookup = minDesign[minDesign['rad_n'] == 1.3].set_index('component_id')['Min K']
        
        # 2. Update the 'K' column for the entire dataframe.
        minDesign['Min K'] = minDesign['component_id'].map(k_13_lookup)
        
        if rad_nb is not None:
            minDesign = minDesign[['component_id','Min Design massFlow','Min K','Min Tout design',"Min TotalHeatloss",'rad_n']].query('rad_n==@rad_nb')
                    
        df_min = pd.merge(heatloss, minDesign, how = 'left', on = 'component_id')
        
    if path_load_max is not None:    
        with open(path_load_max, 'rb') as f:
            maxDesign = pickle.load(f) 
            
        # The nominal K value is based on the nominal radiator exponent which is 1.3
        # 1. Create a lookup table (Series)
        k_13_lookup = maxDesign[maxDesign['rad_n'] == 1.3].set_index('component_id')['Max K']
        
        # 2. Update the 'K' column for the entire dataframe.
        maxDesign['Max K'] = maxDesign['component_id'].map(k_13_lookup)
        
        if rad_nb is not None:
            maxDesign = maxDesign[['component_id','Max Design massFlow','Max K','rad_n']].query('rad_n==@rad_nb')
            
        df_max = pd.merge(heatloss, maxDesign, how = 'left', on = 'component_id')

    if method == "mnorm":
        # Calculate supply and return temperature based on design values from EPC

            df_min['Actual Tsupply'], df_min['Actual Treturn'] = zip(*df_min.apply(
                lambda x: get_supply_return_temperature(
                    x['Min Design massFlow'], 
                    x['Min K'], 
                    x[col_HL],
                    x['Tin'],
                    x['rad_n']), axis=1))
            
            df_min.to_parquet(path_save_min)

            df_max['Actual Tsupply'], df_max['Actual Treturn'] = zip(*df_max.apply(
                lambda x: get_supply_return_temperature(
                    x['Max Design massFlow'], 
                    x['Max K'], 
                    x[col_HL], 
                    x['Tin'],
                    x['rad_n']), axis=1))
            
            df_max.to_parquet(path_save_max)
            
    elif method == "deltaT":
        # Convert "type of system" to float and average
        if path_load_BBR is not None:
            with open(path_load_BBR, 'rb') as f:
                BBR = pickle.load(f) 
        
        BBR["Type of system"] = BBR["Type of system"].apply(itemToFloat)
        BBR["Type of system"] = BBR["Type of system"].apply(lambda x: np.median(x) if isinstance(x, list) else np.nan) 
        
        df_min = df_min.merge(BBR[['component_id','Type of system']], how = 'left')
        df_max = df_max.merge(BBR[['component_id','Type of system']], how = 'left')
        
        # Set the actual, current heatloss
        act_cur_HL = heatloss.query('Tin==20 & Ws==5 & Tout==-12')[['component_id','param_stats','TotalHeatloss']].reset_index(drop=True).rename(columns={'TotalHeatloss':'Actual Design TotalHeatloss'})
        
        df_min = df_min.merge(act_cur_HL, how = 'left')
        df_max = df_max.merge(act_cur_HL, how = 'left')
        
        # Calculate supply and return temperature based on design values from EPC # Set the temperature drop to 20C for double string or 10C for single string 

        df_min['Actual Design massFlow'] = df_min.apply(
            lambda x: get_radiator_design(
                (60 if x['Type of system'] == 2 else 80),
                (40 if x['Type of system'] == 2 else 70),
                20,
                x['Actual Design TotalHeatloss'],
                None,
                getKValue=False), axis=1) 
        
        # Make sure to bound the mass flow 
        df_min['Actual Design massFlow'] = np.where(df_min['Actual Design massFlow']>df_min['Min Design massFlow'],df_min['Min Design massFlow'],df_min['Actual Design massFlow'])
                
        df_min['Actual Tsupply'], df_min['Actual Treturn'] = zip(*df_min.apply(
            lambda x: get_supply_return_temperature(
                x['Actual Design massFlow'], 
                x['Min K'], 
                x[col_HL],
                x['Tin'],
                x['rad_n']), axis=1))
        
        df_min.drop(columns = 'Type of system', inplace = True)
        
        df_min.to_parquet(path_save_min)
    
        df_max['Actual Design massFlow'] = df_max.apply(
            lambda x: get_radiator_design(
                (60 if x['Type of system'] == 2 else 80),
                (40 if x['Type of system'] == 2 else 70),
                20,
                x['Actual Design TotalHeatloss'],
                None,
                getKValue=False), axis=1) 
        
        # Make sure to bound the mass flow 
        df_max['Actual Design massFlow'] = np.where(df_max['Actual Design massFlow']>df_max['Max Design massFlow'],df_max['Max Design massFlow'],df_max['Actual Design massFlow'])
           
        df_max['Actual Tsupply'], df_max['Actual Treturn'] = zip(*df_max.apply(
            lambda x: get_supply_return_temperature(
                x['Actual Design massFlow'], 
                x['Max K'], 
                x[col_HL], 
                x['Tin'],
                x['rad_n']), axis=1))
        
        df_max.drop(columns = 'Type of system', inplace = True)
        
        df_max.to_parquet(path_save_max)
        
    else:
        print("Method doesn't exist - check spelling mnorm or deltaT")

