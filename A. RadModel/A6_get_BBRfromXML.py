# -*- coding: utf-8 -*-
"""
A6_get_BBRfromXML.py

Extracts the BBR (Danish Building and Housing Register) information from the
parsed EPC XML data and converts it into a flat per-building table, grouped by
BFE number, BuildingNumber and EPC LabelStatus.

Where multiple buildings within the same EPC label share the same
BuildingNumber, UseCode and YearOfConstruction, their BBR data is aggregated
into a single row.
"""
import pickle
import pandas as pd
import numpy as np
from itertools import chain

def get_BBRfromXML(path_load_data: str, path_save_data: str) -> None:
    """
    Extract BBR (building registry) data from the EPC XML dataset and save it
    as a flat dataframe.

    path_load_data: str
        File path (.pkl) of the EPC XML dataset (output of A2_get_EPC_XML.py
        or later A-series scripts).
    path_save_data: str
        File path (.pkl) where the resulting BBR dataframe is saved.

    Returns
    -------
    None
        The result is written to `path_save_data` as a pickle file containing
        a pandas DataFrame with one row per building (or aggregated group of
        buildings), including construction year, floor area, heating system
        type, and EPC metadata.
    """


    # Helper to detect "empty" objects similar to original use of .empty
    def _is_empty(x):
        if x is None:
            return True
        if hasattr(x, "empty"):
            try:
                return x.empty
            except Exception:
                pass
        if isinstance(x, (dict, list, tuple, set)):
            return len(x) == 0
        return False

    def has_value(xmls):
        validXMLs = [x for x in xmls if isinstance(x, dict)]
        return bool(validXMLs) and not all(x is None for x in validXMLs)

    # LOAD DATA
    with open(path_load_data, "rb") as f:
        building_XML = pickle.load(f)

    rows = []

    for label in building_XML:
        labelWithinRow = []

        # label metadata (single row values)
        label_meta = {
            "BFEnummer": label.get("BFEnummer"),
            "husnummer_ID": label.get("husnummer_ID"),
            "streetName": label.get("streetName"),
            "house_number": label.get("house_number"),
            "EPCSerial": label.get("EPCSerial"),
            "LabelStatus": label.get("LabelStatus"),
            "ValidFrom": label.get("ValidFrom"),
            "ValidTo": label.get("ValidTo"),
            "EPC_Class_Property": label.get("EPC_Class_Property"),
            "lat": label.get("lat"),
            "lon": label.get("lon"),
        }
        
        xmls = label.get("xml", [])
        for xml in xmls:
            row = dict(label_meta)

            if xml is None:
                labelWithinRow.append(row)
                continue
            
            # Type of system
            if not _is_empty(xml.get("heatingsystem")):
                typeSH = xml["heatingsystem"].get("Type of system") if isinstance(xml["heatingsystem"], dict) else xml["heatingsystem"]["Type of system"]
                typeSH = typeSH.tolist()
            else:
                typeSH = np.nan

            # District heating present
            hasDH = [not _is_empty(xml.get("districtheatingsystem"))]

            # Total heated floor area (sum if possible)
            if not _is_empty(xml.get("zones")):
                try:
                    zones = sum(xml["zones"]["TotalHeatedFloorArea"])
                except Exception:
                    try:
                        zones = float(xml["zones"]["TotalHeatedFloorArea"])
                    except Exception:
                        zones = np.nan
            else:
                zones = np.nan
            
            # Add the remaining data (BBR fields + extras)
            bbr_fields = xml.get("BBR", {}) or {}
            if isinstance(bbr_fields, dict):
                row.update(bbr_fields)
            else:
                try:
                    row.update(dict(bbr_fields))
                except Exception:
                    pass

            # extras from xml
            row["Type of system"] = typeSH
            row["hasDH"] = hasDH
            row["TotalHeatedFloorArea"] = zones

            labelWithinRow.append(row)
                
        # Evaluate grouping logic in pure Python
        # Only rows from real xmls count (None-xml rows are pure label_meta
        # copies and never received BBR fields like BuildingNumber)
        real_rows = [r for r in labelWithinRow if 'BuildingNumber' in r]

        def _key(r):
            # normalize NaN -> None so "missing == missing" compares equal
            return tuple(None if pd.isna(v) else v
                         for v in (r.get('BuildingNumber'), r.get('UseCode'),
                                   r.get('YearOfConstruction')))

        all_same = len(real_rows) > 1 and len({_key(r) for r in real_rows}) == 1

        # Tag rows for aggregation and append to master list
        for r in labelWithinRow:
            r['_do_agg'] = all_same
            
        rows.extend(labelWithinRow)
            
    # Create final DataFrame once
    stamdata_BFE = pd.DataFrame(rows)

    # --- Global Aggregation Logic for BBR ---
    def apply_bbr_aggregation(df):
        if df.empty or '_do_agg' not in df.columns:
            return df
        
        df_agg = df[df['_do_agg'] == True].copy()
        df_no_agg = df[df['_do_agg'] == False].copy()
        
        if not df_agg.empty:
            # ALIGNED WITH HEATCALC (Removed geographic and messy metadata)
            group_cols = ['BFEnummer', 'husnummer_ID', 'streetName', 'house_number', 'EPCSerial', 'LabelStatus', 
                          'ValidFrom', 'ValidTo', 'BuildingNumber', 'UseCode', 'YearOfConstruction']
            
            # Ensure all grouping columns exist to prevent KeyError
            for col in group_cols:
                if col not in df_agg.columns: df_agg[col] = None
                
            agg_dict = {
                'Type of system': lambda x: list(set(chain.from_iterable([i if isinstance(i, list) else [] for i in x.dropna()]))),
                'hasDH': lambda x: list({bool(v) for v in chain.from_iterable(x.dropna())}),
                'TotalHeatedFloorArea': 'sum',
                'Total Area': 'sum',
                'DwellingArea': 'sum',
                'CommercialArea': 'sum',
                'lat': 'mean',
                'lon': 'mean',
                'YearOfRenovation': 'max',
                'EPC_Class_Property': lambda x: list(set(x.dropna()))
            }
            
            # Only aggregate columns that actually exist in the dataframe
            valid_agg_dict = {k: v for k, v in agg_dict.items() if k in df_agg.columns}
            
            df_grouped = df_agg.groupby(group_cols, dropna=False).agg(valid_agg_dict).reset_index()
            
            # Realign df_no_agg to match the columns we care about (drops streetName etc. implicitly if not listed)
            # Or just keep everything from the original if you prefer, but let's sync the columns
            # To be safe, we just drop the tag.
            df_final = pd.concat([df_no_agg.drop(columns=['_do_agg']), df_grouped], ignore_index=True)
        else:
            df_final = df_no_agg.drop(columns=['_do_agg'])
            
        return df_final

    # Apply aggregation
    stamdata_BFE = apply_bbr_aggregation(stamdata_BFE)
    
    # Correcting floor area between BBR and EPC (same logic as original: choose the larger)
    # Use max across the two columns, safely handling missing values
    if "Total Area" in stamdata_BFE.columns and "TotalHeatedFloorArea" in stamdata_BFE.columns:
        stamdata_BFE["FloorArea"] = stamdata_BFE[["Total Area", "TotalHeatedFloorArea"]].max(axis=1)
    elif "Total Area" in stamdata_BFE.columns:
        stamdata_BFE["FloorArea"] = stamdata_BFE["Total Area"]
    else:
        stamdata_BFE["FloorArea"] = stamdata_BFE.get("TotalHeatedFloorArea", pd.Series(np.nan, index=stamdata_BFE.index))

    # SAVE DATA
    with open(path_save_data, "wb") as f:
        pickle.dump(stamdata_BFE, f)

