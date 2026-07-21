# -*- coding: utf-8 -*-
"""
A8_get_aggXML.py

Aggregates the per-building EPC tables produced by A6_get_BBRfromXML.py and
A7_get_heatCalcfromXML.py into per-component tables, where a "component" is a
user-defined grouping of one or more buildings (e.g. all buildings sharing a
single district heating substation), as specified in the `compID` mapping
table provided by the user.

Main steps:
    1. get_latest_EPC(): for each component, resolve which EPC label(s) to
       keep when multiple (possibly overlapping or expired) EPCs exist.
    2. aggBBR(): aggregate the BBR (building registry) data per component.
    3. aggCurrent(): aggregate the current-design heat loss per component.
    4. aggMaxMin(): aggregate the minimum/maximum original-design heat loss
       and radiator parameters per component.

get_aggXMLs() runs all of the above steps in sequence.
"""

import pickle
import pandas as pd
import numpy as np
import ast
import math

from A5_get_DHTemperatures import get_supply_return_temperature

def getCmpID4Agg(df: pd.DataFrame, compID: pd.DataFrame) -> pd.DataFrame:
    """
    Add a "component_id" column to `df` by mapping its BFEnummer and EPCSerial
    values to the user-defined component IDs in `compID`.

    df: pd.DataFrame
        Dataframe containing "BFEnummer" and "EPCSerial" columns.
    compID: pd.DataFrame
        User-specified component ID mapping table. Must contain a
        "component_id" column, and may contain "bfe" and/or "epc" columns
        listing the BFE numbers / EPC serials belonging to each component
        (as scalars, lists, or stringified lists).

    Returns
    -------
    pd.DataFrame
        `df` with an added "component_id" column. If a row's BFEnummer and
        EPCSerial map to two different components, the value is set to
        "CONFLICT::<bfe_component>__<epc_component>" for manual inspection.
    """
    
    def _ensure_list(x):
        """Return a list for x that may be list, int, str(list), or NaN."""
        if any(pd.isna(x)):
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, (int, float)):
            return [str(int(x))]
        s = str(x).strip()
        # handle stringified list like "['1001','1002']" or "[1001, 1002]"
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(int(i)) if isinstance(i,(int,float)) else str(i).strip() for i in parsed]
            except Exception:
                pass
        # otherwise single value
        try:
            # attempt numeric -> string without decimals
            if s.isdigit():
                return [s]
            # if it's numeric-like float "1001.0"
            f = float(s)
            if f.is_integer():
                return [str(int(f))]
        except Exception:
            pass
        return [s]

    """Return two dicts: bfe_map and epc_map mapping normalized id -> component_id"""
    bfe_map = {}
    epc_map = {}

    for _, row in compID.iterrows():
        comp = row['component_id']
        # BFEs
        if 'bfe' in row:
            for b in _ensure_list(row['bfe']):
                key = str(b).strip()
                if key:
                    bfe_map[key] = comp
        # EPCs (if present)
        if 'epc' in row:
            for e in _ensure_list(row['epc']):
                key = str(e).strip()
                if key:
                    epc_map[key] = comp

    
    # Component ID for BFEnummer or Husnummer (a safety check, either one should be sufficient but this is for fault detection)
    def choose_component(row):
        # prefer BFE mapping if available; otherwise EPC mapping.
        c_bfe = bfe_map.get(row['BFEnummer'])
        c_epc = epc_map.get(row['EPCSerial'])
        
        if c_bfe and c_epc:
            if c_bfe != c_epc:
                # conflict — record both for inspection
                return f"CONFLICT::{c_bfe}__{c_epc}"
            return c_bfe
        return c_bfe or c_epc  # one of them or None
    
    # apply to your dataframe
    df['component_id'] = df.apply(choose_component, axis=1)
    
    return df   

def cleanDF(df: pd.DataFrame, otherColumns: list = None) -> pd.DataFrame:
    """
    Remove rows with missing EPCSerial values and drop the redundant
    "husnummer_ID" column (BFEnummer is sufficient once EPCs are resolved).

    df: pd.DataFrame
        Input dataframe. Must contain "EPCSerial" and "husnummer_ID" columns.
    otherColumns: list of str, optional
        Currently unused; reserved for future column-specific handling.

    Returns
    -------
    pd.DataFrame
        `df` with rows missing "EPCSerial" removed and "husnummer_ID" dropped.
    """
    # Remove rows with missing EPCs
    df = df.dropna(subset=['EPCSerial'])

    ## We got EPCSerial from BFE and husnr. BFE is parent to husnr, so we can drop husnr.
    ## Meter id and location id has nothing to do with obtaining the EPCSerials. So we can safely remove them.
    ## We can have duplicate of EPCSerial because:
        ## BFE was repeated (because of different husnr id linking to same BFE)
        ## Because husnr got the same EPCs as we got from using the BFE
    
    # Drop meter location ID and husnummer ID as they are not useful in aggregating
    df = df.drop(columns=['husnummer_ID'])#,'meter_ID','location_ID'])
        
    return df

def get_latest_EPC(path_load_data: str, compID: pd.DataFrame) -> pd.DataFrame:
    """
    For each component (as defined in `compID`), resolve which EPC label(s)
    should be kept when multiple, possibly overlapping or expired, EPCs exist
    for the buildings in that component.

    path_load_data: str
        File path (.pkl) of the BBR dataframe (output of A6_get_BBRfromXML.py),
        containing BFE number, husnummer ID, EPCSerial, LabelStatus, validity
        dates and FloorArea per building.
    compID: pd.DataFrame
        User-specified component ID mapping table (see getCmpID4Agg()).

    The function handles duplicate EPCSerials and ensures that the latest EPC
    covering all buildings within a component is kept.

    Returns
    -------
    pd.DataFrame
        A single-column dataframe with column "EPCSerial", listing the EPC
        Serial Identifiers to keep for downstream aggregation.
    """
    
    # Load the dataframe
    with open(path_load_data, 'rb') as f:
        df = pickle.load(f)
    
    # Drop missing EPCs and remove repeated duplicates (same EPCs)
    df = cleanDF(df)
    
    # Add component ID
    df = getCmpID4Agg(df, compID)
    
    # --- Tune the matching algorithm: the upper and bottom threshold for assigning whether floor areas are the same or different
    upThrsh = 0.90
    btmTrsh = 0.50
        
    # ---- 2: Check and flag if we have deviating number of buildings and floor area between EPCSerials for same BFE ----
    
    # Get helper function
    def is_dupl_handle(gNonValid_df, grp, upThrsh = upThrsh, btmTrsh = btmTrsh):
        # Get the duplicates
        gNonValid_dupl = gNonValid_df[gNonValid_df['is_duplicate']]
                
        # Flag the maximum 
        gNonValid_dupl['is_max'] = False 
        gNonValid_dupl.loc[gNonValid_dupl.groupby(grp)['FloorArea'].idxmax(),'is_max'] = True
        
        # Create a new dataframe to avoid problems
        gNonValid_agg = gNonValid_dupl.copy()
        
        # Calculate the ratio between area and max area
        gNonValid_agg['maxm2'] = gNonValid_agg.groupby(grp)['FloorArea'].transform('max')
        gNonValid_agg['m2_ratio'] = round(gNonValid_agg['FloorArea'] / gNonValid_agg['maxm2'],1)
        gNonValid_agg['very_close'] = gNonValid_agg['m2_ratio'] >= upThrsh
        gNonValid_agg['very_far'] = gNonValid_agg['m2_ratio'] < btmTrsh
        gNonValid_agg['btw_m2'] = gNonValid_agg['m2_ratio'].between(btmTrsh, upThrsh, inclusive='left')  
        
        # Keep EPC if very_close is TRUE and latest
        gNonValid_agg['keep'] = False
        
        mask_close = gNonValid_agg['very_close']
        if mask_close.any():
            gNonValid_agg.loc[mask_close, 'keep'] = \
                gNonValid_agg.loc[mask_close].groupby(grp)['ValidFrom'].transform(lambda x: x == x.max())
        
        # Keep the EPC with max area
        mask_far = gNonValid_agg['very_far']
        if mask_far.any():
            gNonValid_agg.loc[mask_far, 'keep'] = \
                gNonValid_agg.loc[mask_far].groupby(grp)['is_max'].transform(lambda x: x == True)
        
        # Keep both if between 0.2 and 0.9
        mask_btw = gNonValid_agg['btw_m2']
        if mask_btw.any():
            gNonValid_agg.loc[mask_btw, 'keep'] = True

        # Get the EPCs to keep
        getEPC = gNonValid_agg[gNonValid_agg['keep']]['EPCSerial'].drop_duplicates().tolist()
        
        return getEPC
    
    def drop_strange_building_numbers(
        g: pd.DataFrame,
        building_col: str = 'BuildingNumber',
        status_col: str = 'LabelStatus',
        valid_statuses=('VALID', 'CORRECTED'),
        min_digits: int = 3,
        proximity: int = 10,
    ):
        import re
        
        """Remove rows whose building number appears only in non-valid rows and
        looks 'strange' (e.g. three-digit 105 when no similar numbers exist in valid rows)."""
        df = g.copy()
        # extract first integer (or NaN)
        df['_bnum'] = df[building_col].astype(str).str.extract(r'(\d+)')[0].astype('Int64')
        valid_nums = df[df[status_col].isin(valid_statuses)]['_bnum'].dropna().astype(int)
        nonvalid_nums = df[~df[status_col].isin(valid_statuses)]['_bnum'].dropna().astype(int)
    
        if nonvalid_nums.empty:
            return g, set()
    
        valid_set = set(valid_nums)
        valid_digit_lengths = set(valid_nums.astype(str).str.len())
    
        removed = set()
        for n in set(nonvalid_nums):
            digits = len(str(abs(int(n))))
            if digits < min_digits:                     # skip short numbers
                continue
            if n in valid_set:                           # exact match exists in valid
                continue
            if digits in valid_digit_lengths:            # same-digit examples exist in valid
                continue
            if any(abs(v - n) <= proximity for v in valid_nums):  # close number exists
                continue
            removed.add(int(n))
    
        if not removed:
            return g, set()
    
        # Remove rows where extracted number is in removed OR the building string contains the number token
        num_pattern = r'\b(' + '|'.join(map(str, removed)) + r')\b'
        mask_remove = df['_bnum'].isin(removed) | df[building_col].astype(str).str.contains(num_pattern, na=False)
        filtered = df.loc[~mask_remove].drop(columns=['_bnum'])
        
        return filtered[g.columns], removed
    
    def resolve_group(g, upThrsh = upThrsh, btmTrsh = btmTrsh):       
        # Groupby columns
        grp = ['streetName','house_number','BuildingNumber','YearOfConstruction','UseCode']
        
        # Copy df
        g = g.copy()
        
        # Remove redundant columns
        g = g.drop(columns=['Type of system','hasDH','lat','lon','EPC_Class_Property'])
                
        # Remove duplicate entries
        col4dupl = g.drop(columns=['BFEnummer']).columns
        g = g.drop_duplicates(subset=col4dupl)
        
        col4dupl = g.drop(columns=['EPCSerial','ValidFrom','ValidTo']).columns
        g = g.drop_duplicates(subset=col4dupl)
        
        # Edit the UseCode to only have the first two digits
        g['UseCode'] = g['UseCode'].astype(str).str[:2] 
        
        # Check if we have strange building number names in NonValid, if so we discard the EPClabels where any one of these are true
        g, removed = drop_strange_building_numbers(g)
        
        # Get valid EPCs
        gValid = g[g['LabelStatus'].isin(['VALID','CORRECTED'])]
        gNonValid = g[~(g['LabelStatus'].isin(['VALID','CORRECTED']))]
  
        ########################################################################
        #                     VALID - CORRECT DECISION RULE                    #
        ########################################################################
        
        if gValid.empty:
            
            # No action - we go directly to EXPIRED decision rule
            print("Go to EXPIRED decision rule")
        
        elif len(gValid) == 1:
            
            # No action - we go directly to EXPIRED decision rule
            print("Go to EXPIRED decision rule")
        
        else: 
            
            # Is there any duplicates?
            gValid_df = gValid.copy()
            gValid_df['is_duplicate'] = gValid_df.duplicated(subset=grp,keep=False)
            
            if any(gValid_df['is_duplicate']):
                # Get the duplicates
                gValid_dupl = gValid_df[gValid_df['is_duplicate']]
                
                # Get the latest EPC
                gValid_dupl['latest'] = gValid_dupl.groupby(grp)['ValidFrom'].transform('max')
                getEPC = gValid_dupl[gValid_dupl['latest'] == gValid_dupl['ValidFrom']]['EPCSerial'].drop_duplicates().tolist()
                
                # Append the remaining EPCs we want to keep
                getEPC = getEPC + gValid_df[~gValid_df['is_duplicate']]['EPCSerial'].drop_duplicates().tolist()
                
            else: 
                
                # Append the EPCs we want to keep
                getEPC = gValid_df['EPCSerial'].drop_duplicates().tolist()

            gValid = gValid[gValid['EPCSerial'].isin(getEPC)]        
            # We go to EXPIRED decision rule
                
        ########################################################################
        #                          EXPIRED DECISION RULE                       #
        ########################################################################                        
                                 
        if gNonValid.empty:
            
            EPCs2Keep = gValid['EPCSerial'].drop_duplicates().tolist()
            
            return list(dict.fromkeys(EPCs2Keep))
        
        elif len(gNonValid) == 1:
            
            # No action - we go directly to MERGER decision rule
            print("Go to MERGER decision rule - ONLY one row in EXPIRED")
                
        else:
            # Is there any duplicates?
            gNonValid_df = gNonValid.copy()
            gNonValid_df['is_duplicate'] = gNonValid_df.duplicated(subset=grp,keep=False)
            
            if any(gNonValid_df['is_duplicate']):
                
                # Flag the maximum 
                gNonValid_df['is_max'] = gNonValid_df['FloorArea'] == gNonValid_df['FloorArea'].max()
                
                # Do we have both is_max and is_duplicate?
                gNonValid_df['is_max_is_duplicate'] = False
                gNonValid_df.loc[(gNonValid_df['is_max'] & gNonValid_df['is_duplicate']), 'is_max_is_duplicate'] = True
                
                if all(gNonValid_df['is_max_is_duplicate']):
                    # If both are max and duplicate, then keep latest
                    gNonValid_df['keep'] = False
                    mask_max_dup = gNonValid_df['is_max_is_duplicate']
                    
                    if mask_max_dup.any():
                        gNonValid_df.loc[mask_max_dup, 'keep'] = \
                            gNonValid_df.loc[mask_max_dup].groupby(grp)['ValidFrom'].transform(lambda x: x == x.max())
                    
                    getEPC = gNonValid_df[gNonValid_df['keep']]['EPCSerial'].drop_duplicates().tolist()
                
                elif any(gNonValid_df['is_max_is_duplicate']):
                    # Then get its EPCSerial and flag all rows for this EPCSerial as is_max TRUE
                    dupmaxEPC = gNonValid_df[gNonValid_df['is_max_is_duplicate']]['EPCSerial'].iloc[0]
                    gNonValid_df.loc[gNonValid_df['EPCSerial'] == dupmaxEPC, 'is_max'] = True
                    
                    # Calculate total area for the true/false is_max
                    gNonValid_agg = gNonValid_df.groupby(['is_max']).agg({'EPCSerial': lambda x: list(x.dropna()),
                                                                          'ValidFrom': lambda x: list(x.dropna()),
                                                                          'FloorArea': 'sum'}).reset_index()
                    
                    # Calculate the ratio between total area and max area
                    gNonValid_agg['maxm2'] = gNonValid_agg['FloorArea'].max()
                    gNonValid_agg['m2_ratio'] = gNonValid_agg['FloorArea'] / gNonValid_agg['maxm2']
                    gNonValid_agg['very_close'] = gNonValid_agg['m2_ratio'] >= upThrsh
                                        
                    # Is the ratios very_close?
                    if all(gNonValid_agg['very_close']):
                        
                        # Get the latest EPC between is_max = TRUE and FALSE
                        maxTrue_ValidFrom = gNonValid_agg[gNonValid_agg['is_max']]['ValidFrom'].iloc[0]
                        maxTrue_ValidFrom = list(dict.fromkeys(maxTrue_ValidFrom))[0]
                        
                        maxFalse_ValidFrom_latest = gNonValid_agg[~gNonValid_agg['is_max']]['ValidFrom']
                        
                        if maxFalse_ValidFrom_latest.empty:
                            # Get EPC for maximum 
                            getEPC = gNonValid_agg[gNonValid_agg['is_max']]['EPCSerial'].drop_duplicates().iloc[0]
                            
                        else: 
                            maxFalse_ValidFrom_latest = maxFalse_ValidFrom_latest.iloc[0]
                            maxFalse_ValidFrom_latest = max(list(dict.fromkeys(maxFalse_ValidFrom_latest)))
                        
                            if maxTrue_ValidFrom >= maxFalse_ValidFrom_latest:
                                # Get EPC for maximum 
                                getEPC = gNonValid_agg[gNonValid_agg['is_max']]['EPCSerial'].drop_duplicates().iloc[0]
    
                            else:
                                # Get EPC for the others
                                getEPC = gNonValid_agg[~gNonValid_agg['is_max']]['EPCSerial'].drop_duplicates().iloc[0]
                                                                    
                    else:
                        
                        # Handle duplicates in expire and get EPC
                        getEPC = is_dupl_handle(gNonValid_df, grp)
                    
                else:
                    
                    # Handle duplicates in expire and get EPC
                    getEPC = is_dupl_handle(gNonValid_df, grp)

                    # Append the remaining EPCs we want to keep
                    getEPC = getEPC + gNonValid_df[~gNonValid_df['is_duplicate']]['EPCSerial'].drop_duplicates().tolist()
 
            else:
                
                # Get the EPCs to keep
                getEPC = gNonValid_df['EPCSerial'].drop_duplicates().tolist()
                            
            gNonValid = gNonValid[gNonValid['EPCSerial'].isin(getEPC)]        
            # We go to EXPIRED decision rule
            
        ########################################################################
        #                           MERGER DECISION RULE                       #
        ########################################################################                
            
        if gValid.empty:
            
            EPCs2Keep = gNonValid['EPCSerial'].drop_duplicates().tolist()
            
            return list(dict.fromkeys(EPCs2Keep))
            
        else:
            
            # Check if the floor area of expired is similar to valid, if so then we remove the expire
            sumAreaValid = sum(gValid['FloorArea'])
            sumAreaNonValid = sum(gNonValid['FloorArea'])
            
            # Areas are similar?
            m2_ratio_Valid = round(sumAreaValid / max([sumAreaValid,sumAreaNonValid]),1)
            m2_ratio_NonValid = round(sumAreaNonValid / max([sumAreaValid,sumAreaNonValid]),1)
            
            very_close_Valid = m2_ratio_Valid >= upThrsh
            very_close_NonValid = m2_ratio_NonValid >= upThrsh
            
            # If floor areas are very close to each other, we likely have the same building, thus exclude EXPIRE.
            
            if not (very_close_Valid and very_close_NonValid):
            
                gBoth = gValid.merge(gNonValid, how = 'outer')
                
                # Change naming of CORRECTED to valid
                gBoth.loc[gBoth['LabelStatus'] == 'CORRECTED', 'LabelStatus'] = 'VALID'
                
                # Is there any duplicates?
                gBoth_df = gBoth.copy()
                gBoth_df['is_duplicate'] = gBoth_df.duplicated(subset=grp,keep=False)
                
                # Check if EXPIRED and VALID EPC match in area (aka proxy for represent same buildings) but only if any building numbers are the same
                builValid = gBoth_df[gBoth_df['LabelStatus'] == 'VALID']['BuildingNumber'].drop_duplicates()
                builNonValid = gBoth_df[~(gBoth_df['LabelStatus'] == 'VALID')]['BuildingNumber'].drop_duplicates()
                
                if any(gBoth_df['is_duplicate']) and any(builValid.isin(builNonValid)):
                    # Get the total area between valid and non-valid
                    gBoth_agg = gBoth_df.groupby(['LabelStatus']).agg({'FloorArea': 'sum'}).reset_index()
                    
                    # Flag the maximum 
                    gBoth_agg['is_max'] = gBoth_agg['FloorArea'] == gBoth_agg['FloorArea'].max()
                    
                    # Now check if the Total Area is very different or very close.  
                    gBoth_agg['maxm2'] = gBoth_agg['FloorArea'].max()
                    gBoth_agg['m2_ratio'] = round(gBoth_agg['FloorArea'] / gBoth_agg['maxm2'],1)
                    gBoth_agg['very_close'] = gBoth_agg['m2_ratio'] >= upThrsh
                    gBoth_agg['very_far'] = gBoth_agg['m2_ratio'] < btmTrsh
                    
                    # Is the ratios very_close?
                    if all(gBoth_agg['very_close']):
                        
                        EPCs2Keep = gValid['EPCSerial'].drop_duplicates().tolist()
                        
                        return list(dict.fromkeys(EPCs2Keep))
                        
                    elif any(gBoth_agg['very_far']):
                        
                        # Get the labelstatus for the maximum EPC 
                        maxTrue_Label = gBoth_agg[gBoth_agg['is_max']]['LabelStatus'].iloc[0]
                        
                        if maxTrue_Label == 'VALID':
                            
                            EPCs2Keep = gValid['EPCSerial'].drop_duplicates().tolist()
                            
                            return list(dict.fromkeys(EPCs2Keep))
        
                        else:
                            
                            EPCs2Keep = gNonValid['EPCSerial'].drop_duplicates().tolist()
                            
                            return list(dict.fromkeys(EPCs2Keep))
                        
                    else:
                        
                        # Handle duplicates in expire and get EPC
                        getEPC = is_dupl_handle(gBoth_df, grp)
    
                        # Append the remaining EPCs we want to keep
                        EPCs2Keep = getEPC + gBoth_df[~gBoth_df['is_duplicate']]['EPCSerial'].drop_duplicates().tolist()
                        
                        return list(dict.fromkeys(EPCs2Keep))
                        
                else:
                    
                    # Get the EPCs to keep
                    EPCs2Keep = gBoth_df['EPCSerial'].drop_duplicates().tolist()
                    
                    return list(dict.fromkeys(EPCs2Keep))
                
            else:
                
                # Get the EPCs to keep
                EPCs2Keep = gValid['EPCSerial'].drop_duplicates().tolist()
                
                return list(dict.fromkeys(EPCs2Keep))
                
           
    compID = compID[['component_id']].drop_duplicates()
    EPCs2Keep = []
    
    for _, r in compID.iterrows():
        
        # For a BFEnummer run resolve_group
        dfsub = df[df['component_id'] == r['component_id']].copy()
        
        EPCs2Keep = EPCs2Keep + resolve_group(dfsub)
        
    return pd.DataFrame(EPCs2Keep, columns=['EPCSerial']) 

def weighted_average(group,x,weight):
    # x is the name of the column we want to apply weighting
    # weight is the name of the column containing the weights
    
    total_weight = group[weight].sum()
    
    if total_weight == 0:  # Handle case where all weights might be zero
        return 0
    
    return (group[x] * group[weight]).sum() / total_weight

def prefix_name_to_column(data, prefix, exclude_columns):
    # exclude_columns = ['Column1', 'Column2']  

    # Create a dictionary for renaming, excluding the specified columns
    renaming_dict = {col: prefix + col for col in data.columns if col not in exclude_columns}

    # Rename the columns using the dictionary
    data = data.rename(columns=renaming_dict)
    
    return data

def epc_weighted_average(group, epc_col='EPC_Class_Property', weight_col='FloorArea', return_label=False):
    """
    Calculates the floor-area weighted average of EPC labels for a given cluster of buildings.
    Safely parses lists/strings/arrays on the fly without mutating the original dataframe.
    
    If return_label=True, converts the final numeric average back to a string grade (A-G).
    """
    mapping = {
        'G2': 'G', 'G1': 'G', 'F2': 'G', 'E2': 'F', 'F1': 'F',
        'D2': 'E', 'E1': 'E', 'C2': 'D', 'D1': 'D', 'B2': 'C', 'C1': 'C',
        'B1': 'B', 'A2': 'A', 'A1': 'A', 'A2010': 'A', 'A2015': 'A', 'A2020': 'A',
        'G': 'G', 'F': 'F', 'E': 'E', 'D': 'D', 'C': 'C', 'B': 'B', 'A': 'A'
    }
    
    scale_map = {'G': 1, 'F': 2, 'E': 3, 'D': 4, 'C': 5, 'B': 6, 'A': 7}
    reverse_map = {1: 'G', 2: 'F', 3: 'E', 4: 'D', 5: 'C', 6: 'B', 7: 'A'}
    
    total_weighted_score = 0.0
    total_valid_weight = 0.0
    
    # Iterate through the rows in this specific group/cluster
    for idx, row in group.iterrows():
        raw_input = row[epc_col]
        weight = row[weight_col]
        
        # 1. Skip if weight is missing or zero
        if pd.isna(weight) or weight <= 0:
            continue
            
        # 2. Skip if EPC is missing entirely
        if raw_input is None or (isinstance(raw_input, float) and np.isnan(raw_input)):
            continue
            
        # 3. Safely parse the EPC input into a list
        epc_list = []
        if isinstance(raw_input, str):
            raw_input = raw_input.strip()
            if raw_input.startswith('[') and raw_input.endswith(']'):
                try:
                    epc_list = ast.literal_eval(raw_input)
                except (ValueError, SyntaxError):
                    epc_list = [raw_input]
            else:
                epc_list = [raw_input]
        elif isinstance(raw_input, (list, np.ndarray, set)):
            epc_list = list(raw_input)
            
        if not epc_list:
            continue
            
        # 4. Convert the list of labels to numeric values
        numeric_values = []
        for epc in epc_list:
            epc_str = str(epc).strip() if epc is not None else ''
            standardized_label = mapping.get(epc_str, epc_str)
            num = scale_map.get(standardized_label)
            if num is not None:
                numeric_values.append(num)
                
        # 5. Calculate row average and apply cluster weight
        if numeric_values:
            row_epc_score = sum(numeric_values) / len(numeric_values)
            total_weighted_score += (row_epc_score * weight)
            total_valid_weight += weight
            
    # 6. Return final result
    if total_valid_weight == 0:
        return np.nan
        
    avg_score = total_weighted_score / total_valid_weight
    
    # 7. Convert back to string if requested
    if return_label:
        # standard "round half up" math to avoid Python's banker's rounding bias
        rounded_score = math.floor(avg_score + 0.5) 
        # keep it safely within the 1-7 bounds
        clamped_score = max(1, min(7, rounded_score)) 
        return reverse_map[clamped_score]
        
    return avg_score

def aggBBR(path_load_data: str, path_save_data: str, EPCs2Keep: pd.DataFrame, compID: pd.DataFrame) -> None:
    """
    Aggregate the BBR (building registry) data per component and save.

    path_load_data: str
        File path (.pkl) of the BBR dataframe (output of A6_get_BBRfromXML.py).
    path_save_data: str
        File path (.pkl) where the aggregated BBR dataframe is saved.
    EPCs2Keep: pd.DataFrame
        Single-column dataframe with column "EPCSerial" (output of get_latest_EPC()).
    compID: pd.DataFrame
        User-specified component ID mapping table (see getCmpID4Agg()).

    Returns
    -------
    None
        The aggregated dataframe is written to `path_save_data` as a pickle
        file, with one row per component_id.
    """
    
    # Aggregate the BBR_XML per BFEnummer and save
    with open(path_load_data, 'rb') as f:
        df = pickle.load(f)
        
    # Drop missing EPCs
    df = cleanDF(df) 
    
    # Get EPCs to keep
    df = df[df['EPCSerial'].isin(EPCs2Keep['EPCSerial'])]
    
    # Make column type hashable (for removing duplicates)
    df['Type of system'] = df['Type of system'].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
    df['EPC_Class_Property'] = df['EPC_Class_Property'].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
    df['hasDH'] = df['hasDH'].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
    
    # Remove duplicates (because of how EPCs were fetched we have duplicates, requires we remove husnummer ID)
    df = df.drop_duplicates()
    
    # Make column type list
    df['Type of system'] = df['Type of system'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    df['EPC_Class_Property'] = df['EPC_Class_Property'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    df['hasDH'] = df['hasDH'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    
    # Prepare column 
    df['YrConList'] = df['YearOfConstruction']
    df['YrRenList'] = df['YearOfRenovation']
    df['UseCodeList'] = df['UseCode']
    df['EPC_Class_List'] = df['EPC_Class_Property']
        
    # Add component ID
    df = getCmpID4Agg(df, compID)
    
    aggregated_df = df.groupby('component_id').agg({
        'BFEnummer': lambda x: list(x.dropna()),
        'EPCSerial': lambda x: list(x.dropna()),
        'LabelStatus': lambda x: list(x.dropna()),
        'ValidFrom': lambda x: list(x.dropna()),
        'ValidTo': lambda x: list(x.dropna()),
        'lat': lambda x: weighted_average(df.loc[x.index],'lat','FloorArea'),
        'lon': lambda x: weighted_average(df.loc[x.index],'lon','FloorArea'),
        'EPC_Class_Property': lambda x: epc_weighted_average(df.loc[x.index],'EPC_Class_Property','FloorArea', return_label=True),
        'EPC_Class_List': lambda x: list(x.dropna()),
        'BuildingNumber': lambda x: list(x.dropna()),
        'UseCode': lambda x: weighted_average(df.loc[x.index],'UseCode','FloorArea'),
        'UseCodeList': lambda x: list(x.dropna()),
        'YearOfConstruction': lambda x: weighted_average(df.loc[x.index],'YearOfConstruction','FloorArea'),
        'YrConList': lambda x: list(x.dropna()),
        'YearOfRenovation': lambda x: weighted_average(df.loc[x.index],'YearOfRenovation','FloorArea'),
        'YrRenList': lambda x: list(x.dropna()),
        'Type of system': 'sum',
        'hasDH': 'sum',
        'TotalHeatedFloorArea': 'sum',
        'Total Area': 'sum',
        'FloorArea': 'sum',
        'DwellingArea': 'sum',
        'CommercialArea': 'sum'
        }).reset_index()
           
    # Set to empty list
    aggregated_df["Type of system"] = aggregated_df["Type of system"].apply(lambda x: [] if x == 0 else x)
    aggregated_df["EPC_Class_List"] = aggregated_df["EPC_Class_List"].apply(lambda x: [] if x == 0 else x)
    
    # Add number of buildings by counting BuildingNumber
    aggregated_df['NumberOfBuildings'] = aggregated_df["BuildingNumber"].apply(lambda x: len(x))
    
    with open(path_save_data, 'wb') as f:
        pickle.dump(aggregated_df, f)  
    
def aggCurrent(path_load_data: str, path_save_data: str, EPCs2Keep: pd.DataFrame, compID: pd.DataFrame) -> None:
    """
    Aggregate the current-design heat loss table per component and save.

    path_load_data: str
        File path (.pkl) of the current-design heat loss table (output of
        A7_get_heatCalcfromXML.py, `path_save_data_current`).
    path_save_data: str
        File path (.pkl) where the aggregated table is saved.
    EPCs2Keep: pd.DataFrame
        Single-column dataframe with column "EPCSerial" (output of get_latest_EPC()).
    compID: pd.DataFrame
        User-specified component ID mapping table (see getCmpID4Agg()).

    Returns
    -------
    None
        The aggregated dataframe is written to `path_save_data` as a pickle
        file, with one row per component_id.
    """
    # Aggregate the currentDesign per BFEnummer
    with open(path_load_data, 'rb') as f:
        df = pickle.load(f) 
        
    # Drop missing EPCs
    df = cleanDF(df) 
    
    # Get only latest EPCs
    df = df[df['EPCSerial'].isin(EPCs2Keep['EPCSerial'])]
    
    # Remove duplicates
    df = df.drop_duplicates()
        
    # Add component ID
    df = getCmpID4Agg(df, compID)
    
    aggregated_df = df.groupby('component_id').agg({
        'BFEnummer': lambda x: list(x.dropna()),
        'EPCSerial': lambda x: list(x.dropna()),
        'LabelStatus': lambda x: list(x.dropna()),
        'BuildingNumber': lambda x: list(x.dropna()),
        'windows': 'sum',
        'buildingparts': 'sum',
        'linelosses': 'sum',
        'ventilation': 'sum', 
        'TotalHeatloss': 'sum',
        'InternalHeatGain': 'sum',
        'HeatlossInclGain': 'sum'
        }).reset_index()
    
    # Add number of buildings by counting BuildingNumber
    aggregated_df['NumberOfBuildings'] = aggregated_df["BuildingNumber"].apply(lambda x: len(x))
    
    with open(path_save_data, 'wb') as f:
        pickle.dump(aggregated_df, f)  

def aggMaxMin(minOrMax: str, path_load_data: str, path_load_data_BBR: str, path_save_data: str,
              EPCs2Keep: pd.DataFrame, compID: pd.DataFrame) -> None:
    """
    Aggregate the minimum or maximum original-design heat loss and radiator
    parameter table per component and save.

    minOrMax: str
        Either "Min" or "Max", selecting which original-design table to
        aggregate, and used as a column-name prefix in the output.
    path_load_data: str
        File path (.pkl) of the minimum or maximum original-design table
        (output of A7_get_heatCalcfromXML.py, `path_save_data_min` or
        `path_save_data_max`).
    path_load_data_BBR: str
        File path (.pkl) of the BBR dataframe (output of A6_get_BBRfromXML.py),
        used to retrieve FloorArea for weighted averages.
    path_save_data: str
        File path (.pkl) where the aggregated table is saved.
    EPCs2Keep: pd.DataFrame
        Single-column dataframe with column "EPCSerial" (output of get_latest_EPC()).
    compID: pd.DataFrame
        User-specified component ID mapping table (see getCmpID4Agg()).

    Returns
    -------
    None
        The aggregated dataframe is written to `path_save_data` as a pickle
        file, with one row per (component_id, rad_n).
    """
    # Aggregate the orgDesign per BFEnummer
    with open(path_load_data, 'rb') as f:
        df = pickle.load(f) 
            
    # Load the BBR data to get floor area
    with open(path_load_data_BBR, 'rb') as f:
        df_BBR = pickle.load(f)
    
    del f
    
    # Drop missing EPCs
    df = cleanDF(df) 
    df_BBR = cleanDF(df_BBR)[['BFEnummer','EPCSerial','LabelStatus','ValidFrom','ValidTo','BuildingNumber','FloorArea']].drop_duplicates()
        
    # Get only latest EPCs
    df = df[df['EPCSerial'].isin(EPCs2Keep['EPCSerial'])]
    
    # Make column type hashable (for removing duplicates)
    df['InclMechanicalVentilationLoss'] = df['InclMechanicalVentilationLoss'].apply(lambda x: bool(x) if isinstance(x, (list, np.ndarray)) else x)

    # Remove duplicates
    df = df.drop_duplicates()

    # Add prefix to column names
    df = prefix_name_to_column(df, minOrMax + ' ', ['BFEnummer','EPCSerial','LabelStatus','ValidFrom','ValidTo','BuildingNumber','rad_n','fs','Tin design','T design supply','T design return'])
    
    # Merge floor area with df
    df = pd.merge(df, df_BBR, how = 'left', on = ['BFEnummer','EPCSerial','LabelStatus','ValidFrom','ValidTo','BuildingNumber'])
    
    # Add component ID
    df = getCmpID4Agg(df, compID)
    
    aggregated_df = df.groupby(['component_id', 'fs']).agg({
        'BFEnummer': lambda x: list(x.dropna()),
        'EPCSerial': lambda x: list(x.dropna()),
        'LabelStatus': lambda x: list(x.dropna()),
        'BuildingNumber': lambda x: list(x.dropna()),
        minOrMax + ' windows': 'sum',
        minOrMax + ' buildingparts': 'sum',
        minOrMax + ' linelosses': 'sum',
        minOrMax + ' ventilation': 'sum',  
        minOrMax + ' TotalHeatloss': 'sum',
        minOrMax + ' InternalHeatGain': 'sum',
        minOrMax + ' HeatlossInclGain': 'sum',
        minOrMax + ' InclMechanicalVentilationLoss': lambda x: list(x.dropna()),
        'T design supply': lambda x: weighted_average(df.loc[x.index],'T design supply','FloorArea'),
        'T design return': lambda x: weighted_average(df.loc[x.index],'T design return','FloorArea'),
        'rad_n': 'mean',
        minOrMax + ' K': 'sum',
        minOrMax + ' Tout design': lambda x: weighted_average(df.loc[x.index],minOrMax + ' Tout design','FloorArea'),
        'Tin design': lambda x: weighted_average(df.loc[x.index],'Tin design', minOrMax + ' TotalHeatloss'),
        minOrMax + ' Design massFlow': 'sum',
        minOrMax + ' Radiator power': 'sum',
        }).reset_index()
    
    # Add number of buildings by counting BuildingNumber
    aggregated_df['NumberOfBuildings'] = aggregated_df["BuildingNumber"].apply(lambda x: len(x))

    with open(path_save_data, 'wb') as f:
        pickle.dump(aggregated_df, f)  
 
def get_aggXMLs(path_bbr_xml: str, compID: pd.DataFrame, path_agg_bbr: str,
                path_current_design: str, path_agg_current: str,
                path_max_design: str, path_agg_max: str,
                path_min_design: str, path_agg_min: str) -> None:
    """
    Run the full Part A aggregation workflow: identify the EPCs to keep per
    component, then aggregate the BBR, current-design and min/max
    original-design tables per component.

    path_bbr_xml: str
        File path (.pkl) of the BBR dataframe (output of A6_get_BBRfromXML.py).
    compID: pd.DataFrame
        User-specified component ID mapping table (see getCmpID4Agg()).
    path_agg_bbr: str
        File path (.pkl) where the aggregated BBR table is saved.
    path_current_design: str
        File path (.pkl) of the current-design heat loss table (output of
        A7_get_heatCalcfromXML.py).
    path_agg_current: str
        File path (.pkl) where the aggregated current-design table is saved.
    path_max_design: str
        File path (.pkl) of the maximum original-design table (output of
        A7_get_heatCalcfromXML.py).
    path_agg_max: str
        File path (.pkl) where the aggregated maximum original-design table is saved.
    path_min_design: str
        File path (.pkl) of the minimum original-design table (output of
        A7_get_heatCalcfromXML.py).
    path_agg_min: str
        File path (.pkl) where the aggregated minimum original-design table is saved.

    Returns
    -------
    None
        Each aggregated table is written to its respective file path.
    """
    # 1. Identify relevant EPCs
    EPCs2Keep = get_latest_EPC(path_load_data = path_bbr_xml, 
                               compID = compID)

    # 2. Aggregate BBR Data
    aggBBR(path_load_data = path_bbr_xml, 
           path_save_data = path_agg_bbr, 
           EPCs2Keep = EPCs2Keep, 
           compID = compID)

    # 3. Aggregate Current Design
    aggCurrent(path_load_data = path_current_design, 
               path_save_data = path_agg_current, 
               EPCs2Keep = EPCs2Keep, 
               compID = compID)

    # 4. Aggregate upper and lower nominal values
    aggMaxMin(minOrMax = "Max", 
              path_load_data = path_max_design, 
              path_load_data_BBR = path_bbr_xml, 
              path_save_data = path_agg_max, 
              EPCs2Keep = EPCs2Keep, 
              compID = compID)

    aggMaxMin(minOrMax = "Min", 
              path_load_data = path_min_design, 
              path_load_data_BBR = path_bbr_xml, 
              path_save_data = path_agg_min, 
              EPCs2Keep = EPCs2Keep, 
              compID = compID)