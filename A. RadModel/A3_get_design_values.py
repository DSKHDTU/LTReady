# -*- coding: utf-8 -*-
"""
A3_get_design_values.py

Assigns historic U-values and supply/return temperature sets to each building
based on its construction year.

Temperature sets are based on Oestergaard, Energy and Buildings, 126:375-383, 2016
and DS469. U-values are based on historical Danish building regulations and
typical construction practice for each construction period (Byggeskik.dk and
J. Engelmark's work).

For each building part (windows, exterior walls, floors, roofs, etc.), a minimum
and maximum typical U-value is assigned based on the construction year, and
compared against the U-value reported in the EPC to determine the design U-value
range used for heat loss calculations (A4_get_heatloss.py).
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

def get_design_values(path_load_data: str, path_save_data: str) -> None:
    """
    Assign historic design U-values to each building part in the EPC dataset,
    based on the building's construction year.

    path_load_data: str
        File path (.pkl) of the EPC XML dataset (output of A2_get_EPC_XML.py).
    path_save_data: str
        File path (.pkl) where the enriched dataset is saved. Can be the same
        as path_load_data to overwrite in place.

    Returns
    -------
    None
        The result is written to `path_save_data` as a pickle file.
    """
    # Assumptions about historical values
    now_yr = datetime.now().year
    
    # Mean and standard deviation for ext. wall, floor, roof and windows and doors with glazing
    typical_uvalues = {
        (1700, 1929):      {'Exterior wall': {'min': 1.08,'max': 1.87},    'Floor': {'min': 0.49, 'max': 1.96},  'Roof': {'min': 0.78, 'max': 1.47},   'Window': {'min': 4.10, 'max': 4.7}},
        (1930, 1960):      {'Exterior wall': {'min': 1.37,'max': 1.46},    'Floor': {'min': 0.49, 'max': 1.96},  'Roof': {'min': 0.78, 'max': 1.47},   'Window': {'min': 2.00, 'max': 2.8}},
        (1961, 1976):      {'Exterior wall': {'min': 0.43,'max': 1.30},    'Floor': {'min': 0.34, 'max': 0.60},  'Roof': {'min': 0.34, 'max': 0.45},   'Window': {'min': 2.00, 'max': 2.8}},
        (1977, 2008):      {'Exterior wall': {'min': 0.20,'max': 0.40},    'Floor': {'min': 0.20, 'max': 0.40},  'Roof': {'min': 0.15, 'max': 0.20},   'Window': {'min': 1.50, 'max': 2.8}},
        (2009, now_yr):    {'Exterior wall': {'min': 0.30,'max': 0.40},    'Floor': {'min': 0.20, 'max': 0.40},  'Roof': {'min': 0.20, 'max': 0.25},   'Window': {'min': 0.80, 'max': 1.5}}
    }
    
    # Upload stamdata
    with open(path_load_data, 'rb') as f:
        building_XML = pickle.load(f)
        
    del f
            
    # Assigning the design values to the building based on its construction year
    
    for label in building_XML: 
        
        xmls = label["xml"]
        
        for xml in xmls:

            if xml is None:
                continue
                
            windows = xml["windows"]
            buildingparts = xml["buildingparts"]

            # --- SKIP if already processed ---
            already_processed_cols = [
                'Byggeskik Uvalue Min',
                'Byggeskik Uvalue Max',
                'Design Uvalue Min',
                'Design Uvalue Max'
            ]
            
            if all(col in windows.columns for col in already_processed_cols) and \
               all(col in buildingparts.columns for col in already_processed_cols):
                continue   # skip this xml, already processed

            # PLACE IN THE CORRECT YEAR RANGE
            for years, values in typical_uvalues.items():
                start, end = years
                if start <= xml["BBR"]["YearOfConstruction"] <= end:
                    typical_uvalue = values
                    break
        
            # DEFINE TYPICAL U VALUES FOR THE BUILDING PARTS
            SEEB = {
                '1-1-0-0': ['Roof and ceiling', 'Roof', typical_uvalue['Roof']['min'], typical_uvalue['Roof']['max']],
                '1-1-1-0': ['Ceiling', 'Roof', typical_uvalue['Roof']['min'], typical_uvalue['Roof']['max']],
                '1-1-2-0': ['Flat Roof', 'Roof', typical_uvalue['Roof']['min'], typical_uvalue['Roof']['max']],
                '1-1-3-0': ['Attic space', 'Roof', typical_uvalue['Roof']['min'], typical_uvalue['Roof']['max']],
                '1-2-0-0': ['Exterior walls', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-2-1-0': ['Cavity exterior walls', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-2-2-0': ['Solid exterior walls', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-2-3-0': ['Light exterior walls', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-2-1-1': ['Cavity walls towards unheated space', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-2-2-1': ['Solid walls towards unheated space', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-2-3-1': ['Light walls towards unheated space', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-2-4-0': ['Basement exterior walls', 'Exterior wall', typical_uvalue['Exterior wall']['min'], typical_uvalue['Exterior wall']['max']],
                '1-3-0-0': ['Windows, skylights, and doors', 'Window', typical_uvalue['Window']['min'], typical_uvalue['Window']['max']],
                '1-3-1-0': ['Windows', 'Window', typical_uvalue['Window']['min'], typical_uvalue['Window']['max']],
                '1-3-2-0': ['Skylights', 'Window', typical_uvalue['Window']['min'], typical_uvalue['Window']['max']],
                '1-3-3-0': ['Exterior doors', 'Window', typical_uvalue['Window']['min'], typical_uvalue['Window']['max']],
                '1-4-0-0': ['Floors', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-1-0': ['Ground floor slab', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-2-0': ['Intermediate floor', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-3-0': ['Crawl space', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-4-0': ['Basement floor', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-1-1': ['Ground floor slab with underfloor heating', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-2-1': ['Intermediate floor with underfloor heating', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-3-1': ['Crawl space with underfloor heating', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-4-1': ['Basement floor with underfloor heating', 'Floor', typical_uvalue['Floor']['min'], typical_uvalue['Floor']['max']],
                '1-4-5-0': ['Line Loss', 'Lineloss', None, None],
            }

            # CREATE DATAFRAME
            columns_class = ['ClassificationDescriptionENG', 'Category', 'Byggeskik Uvalue Min', 'Byggeskik Uvalue Max']
            
            classification = pd.DataFrame.from_dict(SEEB, orient='index', columns=columns_class)
        
            # MERGE DATAFRAMES BASED ON CLASSIFICATION
            if not windows.empty: 
                windows = windows.merge(classification, left_on='Classification', right_index=True, how='left')
                
                # USE LARGEST U VALUE BETWEEN U VALUE FROM EPC AND TYPICAL U VALUE 
                windows['Design Uvalue Min'] = np.where(
                    windows['Uvalue'] > windows['Byggeskik Uvalue Min'],
                    windows['Uvalue'], 
                    windows['Byggeskik Uvalue Min'])
                
                windows['Design Uvalue Max'] = np.where(
                    windows['Uvalue'] > windows['Byggeskik Uvalue Max'],
                    windows['Uvalue'], 
                    windows['Byggeskik Uvalue Max'])
                
                xml["windows"] = windows
            
            if not buildingparts.empty:
                buildingparts = buildingparts.merge(classification, left_on='Classification', right_index=True, how='left')
                
                # USE LARGEST U VALUE BETWEEN U VALUE FROM EPC AND TYPICAL U VALUE
                buildingparts['Design Uvalue Min'] = np.where(
                    buildingparts['Uvalue'] > buildingparts['Byggeskik Uvalue Min'],
                    buildingparts['Uvalue'], 
                    buildingparts['Byggeskik Uvalue Min'])
                
                buildingparts['Design Uvalue Max'] = np.where(
                    buildingparts['Uvalue'] > buildingparts['Byggeskik Uvalue Max'],
                    buildingparts['Uvalue'], 
                    buildingparts['Byggeskik Uvalue Max'])
                
                xml["buildingparts"] = buildingparts
    
    del windows, buildingparts, classification, typical_uvalue, start, end, label, xmls, xml, years, values, SEEB
        
    # Save XML
    
    with open(path_save_data, 'wb') as f:
        pickle.dump(building_XML, f) 




