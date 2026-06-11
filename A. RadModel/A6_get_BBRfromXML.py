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
            #"meter_ID": label.get("meter_ID"),
            "EPCSerial": label.get("EPCSerial"),
            #"location_ID": label.get("location_ID"),
            "LabelStatus": label.get("LabelStatus"),
            "ValidFrom": label.get("ValidFrom"),
            "ValidTo": label.get("ValidTo"),
            "EPC_Class_Property": label.get("EPC_Class_Property"),
            "lat": label.get("lat"),
            "lon": label.get("lon"),
        }
        
        xmls = label.get("xml", [])
        for xml in xmls:
            # Build combined row (label metadata)
            row = {}
            row.update(label_meta)

            if xml is None:
                # just append a label-only row to each list
                rows.append(row)
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
                    # fallback: try to coerce to a numeric scalar if sum fails
                    try:
                        zones = float(xml["zones"]["TotalHeatedFloorArea"])
                    except Exception:
                        zones = np.nan
            else:
                zones = np.nan
            
            # Add the remaining data (BBR fields + extras)
            # xml['BBR'] originally converted to a DataFrame row; assume it's dict-like
            bbr_fields = xml.get("BBR", {}) or {}
            if isinstance(bbr_fields, dict):
                row.update(bbr_fields)
            else:
                # if not dict-like, attempt to coerce to dict (keeps behavior tolerant)
                try:
                    row.update(dict(bbr_fields))
                except Exception:
                    pass

            # extras from xml
            row["Type of system"] = typeSH
            row["hasDH"] = hasDH
            row["TotalHeatedFloorArea"] = zones

            labelWithinRow.append(row)
                
        if has_value(xmls):
            
            if len(labelWithinRow)>1:
                # DataFrame for per-label processing
                label_df = pd.DataFrame(labelWithinRow)    
                        
                # Check if BuildingNumber is unique in the label - if not then we aggregate together
                all_same = label_df[['BuildingNumber','UseCode','YearOfConstruction']].nunique(dropna=False).eq(1).all()
                
                if all_same:
                    agg = label_df.groupby(['BFEnummer', 'husnummer_ID', 'EPCSerial','streetName','house_number', #'location_ID', 'meter_ID', 
                                                         'LabelStatus', 'ValidFrom', 'ValidTo', 'EPC_Class_Property', 'lat','lon',
                                                         'BuildingNumber', 'UseCode', 'YearOfConstruction','YearOfRenovation'], dropna=False).agg({
                                                             'Type of system': lambda x: list(set(chain.from_iterable(x.dropna()))),
                                                             'hasDH': lambda x: list({bool(v) for v in x.dropna()}),
                                                             'TotalHeatedFloorArea': 'sum',
                                                             'Total Area': 'sum',
                                                             'DwellingArea': 'sum',
                                                             'CommercialArea': 'sum'
                                                             }).reset_index()
                    
                    # Convert the entire DataFrame into a list of dictionaries and Use extend to add all items from new_records into rows
                    rows.extend(agg.to_dict(orient='records'))
                    
                else:
                    
                    rows.extend(labelWithinRow)

            else:
                
                rows.extend(labelWithinRow)
                
        else:

            rows.append(row)
            
    # Create final DataFrame once (outside the label loop)
    stamdata_BFE = pd.DataFrame(rows)

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

