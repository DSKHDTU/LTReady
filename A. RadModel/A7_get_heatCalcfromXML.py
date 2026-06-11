# -*- coding: utf-8 -*-
"""
A7_get_heatCalcfromXML.py

Extracts the design condition and radiator parameter information from the EPC
XML dataset (output of A5_get_DHTemperatures.py) and converts it into three
flat per-building tables: current design, maximum original design, and minimum
original design - grouped by BFE number, BuildingNumber and EPC LabelStatus.

Where multiple buildings within the same EPC label share the same
BuildingNumber, UseCode and YearOfConstruction, their values are aggregated
(summed for heat loss components, averaged for temperatures).
"""
import pickle
import pandas as pd
from itertools import zip_longest

def get_heatCalcfromXML_optimized(path_load_data: str,
                                 path_save_data_current: str,
                                 path_save_data_max: str,
                                 path_save_data_min: str,
                                 show_progress: bool = False) -> tuple:
    """
    Extract current, maximum and minimum design heat loss and radiator
    parameter tables from the EPC XML dataset, and save each to a separate file.

    path_load_data: str
        File path (.pkl) of the EPC XML dataset enriched with radiator design
        values (output of A5_get_DHTemperatures.py).
    path_save_data_current: str
        File path (.pkl) where the current-design heat loss table is saved.
    path_save_data_max: str
        File path (.pkl) where the maximum original-design table is saved.
    path_save_data_min: str
        File path (.pkl) where the minimum original-design table is saved.
    show_progress: bool, optional
        If True, prints progress every 1000 processed EPC labels. Default False.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (currentDesign_BFE, originalDesignMax_BFE, originalDesignMin_BFE),
        which are also written to the three output file paths.
    """

    # --- Helper functions ---
    def has_value(xmls):
        validXMLs = [x for x in xmls if isinstance(x, dict)]
        return bool(validXMLs) and not all(x is None for x in validXMLs)

    def py_merge_temp_rad(temps, rads):
        """Pure Python equivalent to the pd.merge(..., on='n') with fallback"""

        # Helper to safely force incoming data into a list of dictionaries
        def _ensure_list(data):
            if isinstance(data, pd.DataFrame):
                return data.to_dict('records')
            if isinstance(data, list):
                return data
            return [] # Fallback for None, NaN, etc.

        temps = _ensure_list(temps)
        rads = _ensure_list(rads)
        
        # Safe check for empty lists
        if len(temps) == 0 and len(rads) == 0:
            return []
        
        has_n_temps = any('n' in t for t in temps if isinstance(t, dict))
        has_n_rads = any('n' in r for r in rads if isinstance(r, dict))
        
        res = []
        # Attempt Inner Join on 'n'
        if has_n_temps and has_n_rads:
            rads_by_n = {}
            for r in rads:
                if isinstance(r, dict) and 'n' in r:
                    rads_by_n.setdefault(r['n'], []).append(r)
            
            for t in temps:
                if isinstance(t, dict):
                    n = t.get('n')
                    if n in rads_by_n:
                        for r in rads_by_n[n]:
                            res.append({**t, **r})
                        
        # Fallback to concat axis=1 equivalent if merge failed/empty
        if not res:
            for t, r in zip_longest(temps, rads, fillvalue={}):
                t_dict = t if isinstance(t, dict) else {}
                r_dict = r if isinstance(r, dict) else {}
                res.append({**t_dict, **r_dict})
                
        # Deduplicate (equivalent to .drop_duplicates())
        seen = set()
        dedup_res = []
        for row in res:
            # Create a hashable representation to check duplicates (ignoring nested lists/dicts if any)
            tup = tuple(sorted((k, str(v)) for k, v in row.items() if not isinstance(v, (list, dict))))
            if tup not in seen:
                seen.add(tup)
                dedup_res.append(row)
        return dedup_res

    # --- Load pickle ---
    with open(path_load_data, 'rb') as f:
        building_XML = pickle.load(f)

    # Containers for rows (dicts)
    current_rows, max_rows, min_rows = [], [], []
    total_labels = len(building_XML)
    label_idx = 0

    for label in building_XML:
        current_labelWithinRow = []
        max_labelWithinRow = []
        min_labelWithinRow = []
        
        label_idx += 1
        
        label_uid = {
            'BFEnummer': label.get('BFEnummer'),
            'husnummer_ID': label.get('husnummer_ID'),
            'EPCSerial': label.get('EPCSerial'),
            'LabelStatus': label.get('LabelStatus'),
            'ValidFrom': label.get('ValidFrom'),
            'ValidTo': label.get('ValidTo'),
        }

        xmls = label.get("xml", [])
        
        for xml in xmls:
            if xml is None:
                current_labelWithinRow.append(dict(label_uid))
                max_labelWithinRow.append(dict(label_uid))
                min_labelWithinRow.append(dict(label_uid))
                continue

            dc = xml.get('DesignCondition', {})
            df0 = dc.get('Design Heatloss Current', {})
            base_current = {k: v for k, v in df0.items() if k not in ('Temperatures - Design Max', 'Temperatures - Design Min')}
            
            bbr = xml.get('BBR') or {}
            building_number_val = bbr.get('BuildingNumber')
            building_useCode_val = bbr.get('UseCode')
            building_yrCon_val = bbr.get('YearOfConstruction')
            
            current_row = dict(label_uid)
            current_row.update({
                'BuildingNumber': building_number_val,
                'UseCode': building_useCode_val,
                'YearOfConstruction': building_yrCon_val
            })
            current_row.update(base_current)
            current_labelWithinRow.append(current_row)

            df1 = dc.get('Design Heatloss Max', {})
            df2 = dc.get('Design Heatloss Min', {})

            keys_exclude = {'Design Heatloss Current', 'Design Heatloss Max', 'Design Heatloss Min'}
            tmp_dict = {k: v for k, v in dc.items() if k not in keys_exclude}

            temps_max = df0.get('Temperatures - Design Max', [])
            temps_min = df0.get('Temperatures - Design Min', [])
            radparam_max = df1.get('Radiator parameter', [])
            radparam_min = df2.get('Radiator parameter', [])

            # Pure Python Merge
            rad1_dicts = py_merge_temp_rad(temps_max, radparam_max)
            rad2_dicts = py_merge_temp_rad(temps_min, radparam_min)

            df1_dict = {k: v for k, v in df1.items() if k != 'Radiator parameter'}
            df2_dict = {k: v for k, v in df2.items() if k != 'Radiator parameter'}

            # Max Processing
            if rad1_dicts:
                for rr in rad1_dicts:
                    max_row = dict(label_uid)
                    max_row.update({
                        'BuildingNumber': building_number_val,
                        'UseCode': building_useCode_val,
                        'YearOfConstruction': building_yrCon_val
                    })
                    max_row.update(rr)
                    max_row.update(tmp_dict)
                    max_row.update(df1_dict)
                    max_labelWithinRow.append(max_row)
            else:
                fallback = dict(label_uid)
                fallback.update({
                    'BuildingNumber': building_number_val,
                    'UseCode': building_useCode_val,
                    'YearOfConstruction': building_yrCon_val
                })
                fallback.update(tmp_dict)
                fallback.update(df1_dict)
                max_labelWithinRow.append(fallback)

            # Min Processing
            if rad2_dicts:
                for rr in rad2_dicts:
                    min_row = dict(label_uid)
                    min_row.update({
                        'BuildingNumber': building_number_val,
                        'UseCode': building_useCode_val,
                        'YearOfConstruction': building_yrCon_val
                    })
                    min_row.update(rr)
                    min_row.update(tmp_dict)
                    min_row.update(df2_dict)
                    min_labelWithinRow.append(min_row)
            else:
                fallback = dict(label_uid)
                fallback.update({
                    'BuildingNumber': building_number_val,
                    'UseCode': building_useCode_val,
                    'YearOfConstruction': building_yrCon_val
                })
                fallback.update(tmp_dict)
                fallback.update(df2_dict)
                min_labelWithinRow.append(fallback)

        # Evaluate grouping logic in pure Python
        all_same = False
        if has_value(xmls) and len(current_labelWithinRow) > 1:
            unique_bld = { (r.get('BuildingNumber'), r.get('UseCode'), r.get('YearOfConstruction')) for r in current_labelWithinRow }
            if len(unique_bld) == 1:
                all_same = True

        # Tag rows for aggregation and append to master list
        for r in current_labelWithinRow: r['_do_agg'] = all_same
        current_rows.extend(current_labelWithinRow)
        
        for r in max_labelWithinRow: r['_do_agg'] = all_same
        max_rows.extend(max_labelWithinRow)
        
        for r in min_labelWithinRow: r['_do_agg'] = all_same
        min_rows.extend(min_labelWithinRow)

        if show_progress and (label_idx % 1000 == 0 or label_idx == total_labels):
            print(f'Processed labels: {label_idx}/{total_labels}')

    # --- Build DataFrames once ---
    df_curr = pd.DataFrame.from_records(current_rows)
    df_max = pd.DataFrame.from_records(max_rows)
    df_min = pd.DataFrame.from_records(min_rows)

    # --- Global Aggregation Logic ---
    def apply_aggregation(df, group_cols, agg_dict):
        if df.empty or '_do_agg' not in df.columns:
            return df
        
        df_agg = df[df['_do_agg'] == True].copy()
        df_no_agg = df[df['_do_agg'] == False].copy()
        
        if not df_agg.empty:
            for col in group_cols:
                if col not in df_agg.columns: df_agg[col] = None
                
            valid_agg_dict = {k: v for k, v in agg_dict.items() if k in df_agg.columns}
            df_grouped = df_agg.groupby(group_cols, dropna=False).agg(valid_agg_dict).reset_index()
            
            df_final = pd.concat([df_no_agg.drop(columns=['_do_agg']), df_grouped], ignore_index=True)
        else:
            df_final = df_no_agg.drop(columns=['_do_agg'])
            
        return df_final

    curr_group_cols = ['BFEnummer', 'husnummer_ID', 'EPCSerial', 'LabelStatus', 'ValidFrom', 'ValidTo', 'BuildingNumber', 'UseCode', 'YearOfConstruction']
    max_min_group_cols = curr_group_cols[:6] + ['n'] + curr_group_cols[6:]

    curr_agg_dict = {
        'windows': 'sum', 'buildingparts': 'sum', 'linelosses': 'sum', 'ventilation': 'sum', 
        'TotalHeatloss': 'sum', 'InternalHeatGain': 'sum', 'HeatlossInclGain': 'sum'
    }

    max_min_agg_dict = curr_agg_dict.copy()
    max_min_agg_dict.update({
        'InclMechanicalVentilationLoss': lambda x: list({bool(v) for v in x.dropna()}),
        'T design supply': 'mean', 'T design return': 'mean', 'expected n': 'mean',
        'K': 'sum', 'Tout design': 'mean', 'Tin design': 'mean', 'Design massFlow': 'sum',
        'Tsupply': 'mean', 'Treturn': 'mean'
    })

    currentDesign_BFE = apply_aggregation(df_curr, curr_group_cols, curr_agg_dict)
    originalDesignMax_BFE = apply_aggregation(df_max, max_min_group_cols, max_min_agg_dict)
    originalDesignMin_BFE = apply_aggregation(df_min, max_min_group_cols, max_min_agg_dict)

    # --- Save to pickle ---
    currentDesign_BFE.to_pickle(path_save_data_current)
    originalDesignMax_BFE.to_pickle(path_save_data_max)
    originalDesignMin_BFE.to_pickle(path_save_data_min)

    return currentDesign_BFE, originalDesignMax_BFE, originalDesignMin_BFE

