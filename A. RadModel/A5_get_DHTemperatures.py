# -*- coding: utf-8 -*-
"""
A5_get_DHTemperatures.py

Calculates the nominal radiator parameters (heat loss coefficient K and design
mass flow) and the corresponding minimum supply and return temperatures for
each building, based on the EPC-derived design heat loss (A4_get_heatloss.py).

Historical supply/return temperature sets before 1960 and until 1998 are based
on Oestergaard, 2016, Energy and Buildings. Values after 1998 are based on
DS469:2007 and DS469:2013.

The water property constants (c_water, rho_water) are averaged over the
temperature range 40-90 degC.

This module also provides get_radiator_design() and get_supply_return_temperature(),
which are reused by A8_get_aggXML.py and C1_get_actualDHTemperatures.py.
"""

import pickle
import numpy as np
from datetime import datetime
import pandas as pd
import copy

def get_radiator_design(Tsupply: float, Treturn: float, Ti: float, Power: float, n,
                         c_water: float = 4189, rho_water: float = 980, getKValue: bool = True):
    """
    Calculate either the radiator heat loss coefficient K or the design mass
    flow, based on the design supply/return/indoor temperatures and design heat loss.

    Tsupply: float
        Design supply temperature [degC].
    Treturn: float
        Design return temperature [degC].
    Ti: float
        Design indoor temperature [degC].
    Power: float
        Design heat loss [W].
    n: float or None
        Radiator exponent. Required if getKValue=True, ignored if False.
    c_water: float, optional
        Specific heat capacity of water [J/(kg*K)]. Default 4189.
    rho_water: float, optional
        Density of water [kg/m3]. Default 980.
    getKValue: bool, optional
        If True (default), returns the radiator heat loss coefficient K [W/K^n].
        If False, returns the design mass flow [kg/s].

    Returns
    -------
    float
        Either the radiator heat loss coefficient K, or the design mass flow,
        depending on `getKValue`.
    """
    
    if getKValue:
        # Calculate the design logaritmic temperature
        LMTD = (Tsupply - Treturn) / np.log((Tsupply - Ti)/(Treturn - Ti))
        
        # Calculate the radiator heat loss coefficient K (UA) [W/m2K] based on design condition
        value = Power / LMTD**n
    else:
        # Calculate the mass flow based on design condition
        value = Power / (c_water*rho_water*(Tsupply - Treturn))
    
    return value

def get_supply_return_temperature(m: float, K: float, Power: float, Ti: float, n: float,
                                   deltaT: float = None, c_water: float = 4189, rho_water: float = 980):
    """
    Calculate the supply and return temperature required to deliver `Power`
    through a radiator with heat loss coefficient `K` and exponent `n`,
    given a mass flow `m` and indoor temperature `Ti`.

    m: float
        Mass flow [kg/s].
    K: float
        Radiator heat loss coefficient [W/K^n].
    Power: float
        Heat demand to be delivered [W].
    Ti: float
        Indoor temperature [degC].
    n: float
        Radiator exponent.
    deltaT: float, optional
        Fixed temperature drop (supply - return) [degC]. If None (default),
        the temperature drop is derived from the mass flow and heat demand.
    c_water: float, optional
        Specific heat capacity of water [J/(kg*K)]. Default 4189.
    rho_water: float, optional
        Density of water [kg/m3]. Default 980.

    Returns
    -------
    tuple of (float, float) or (None, None)
        (Tsupply, Treturn) in degC, or (None, None) if the calculation fails
        (e.g. due to invalid input values).
    """
    
    try:
        
        if deltaT is None:
            # Get the delta temperature
            deltaT = Power /(m*c_water*rho_water)
        
        # Get the supply temperature
        Tsupply = deltaT * np.exp(deltaT * (K / Power)**(1 / n)) / (np.exp(deltaT * (K / Power)**(1 / n)) - 1) + Ti
        
        # Get the return temperature
        Treturn = Tsupply - deltaT
    
    except:
        return None, None
    
    return Tsupply, Treturn

def calc_design_values(xml,getDesignRange,rad_n):    
    K_values = {"n": rad_n + [xml["DesignCondition"]["expected n"]],
                "K": []}

    K_values["K"].extend([None] * len(K_values["n"]))                   
                                   
    K_values = pd.DataFrame(K_values)
    
    Tsupply = xml["DesignCondition"]["T design supply"] 
    Treturn = xml["DesignCondition"]["T design return"] 
    Ti = xml["DesignCondition"]["Tin design"]
    
    if getDesignRange == "Min":
        Power = xml["DesignCondition"]["Design Heatloss Min"]["TotalHeatloss"]
    elif getDesignRange == "Max":
        Power = xml["DesignCondition"]["Design Heatloss Max"]["TotalHeatloss"]
        
    K_values['K'] = K_values.apply(lambda row: get_radiator_design(Tsupply, Treturn, Ti, Power, row['n']), axis = 1)
    m = get_radiator_design(Tsupply, Treturn, Ti, Power, n=None, getKValue=False)
    
    return K_values, m
    
def calc_temperature_values(xml,getDesignRange):
    Power = xml["DesignCondition"]["Design Heatloss Current"]["TotalHeatloss"]
    Ti = xml["DesignCondition"]["Tin design"]
    
    if getDesignRange == "Min":
        m = xml["DesignCondition"]["Design Heatloss Min"]["Design massFlow"]
        rad_n = xml["DesignCondition"]["Design Heatloss Min"]["Radiator parameter"]["n"]
        K = xml["DesignCondition"]["Design Heatloss Min"]["Radiator parameter"]["K"]
        
    elif getDesignRange == "Max":
        m = xml["DesignCondition"]["Design Heatloss Max"]["Design massFlow"]
        rad_n = xml["DesignCondition"]["Design Heatloss Max"]["Radiator parameter"]["n"]
        K = xml["DesignCondition"]["Design Heatloss Max"]["Radiator parameter"]["K"]
        
    temp_values = {"n": rad_n,
                   "K": K,
                   "Tsupply": [],
                   "Treturn": []}
        
    temp_values["Tsupply"].extend([None] * len(temp_values["n"]))
    temp_values["Treturn"].extend([None] * len(temp_values["n"])) 
    
    temp_values = pd.DataFrame(temp_values)
    
    values = temp_values.apply(lambda row: get_supply_return_temperature(m, row['K'], Power, Ti, row['n']) , axis = 1)
    temp_values['Tsupply'], temp_values['Treturn'] = zip(*values)
    
    temp_values = temp_values.drop('K', axis=1)
    
    return temp_values


def get_DHTemperatures(path_load_data: str, path_save_data: str) -> None:
    """
    Calculate the nominal radiator parameters (K, design mass flow) and the
    minimum supply/return temperatures for each building, and save the result.

    path_load_data: str
        File path (.pkl) of the EPC XML dataset enriched with design heat loss
        (output of A4_get_heatloss.py).
    path_save_data: str
        File path (.pkl) where the enriched dataset is saved. Can be the same
        as path_load_data to overwrite in place.

    Returns
    -------
    None
        The result is written to `path_save_data` as a pickle file. Adds
        radiator parameters and "Temperatures - Design Min"/"Temperatures -
        Design Max" entries to each building's "DesignCondition" dictionary.
    """
    # Upload stamdata
    with open(path_load_data, 'rb') as f:
        building_XML = pickle.load(f)
        
    del f

    # Assumptions about historical values
    now_yr = datetime.now().year

    # Radiator exponent range
    n = [1.3]

    # SUPPLY AND RETURN TEMPERATURES
    typical_supplyreturn = {
        (1700, 1960):       {'Supply temperature': 90,     'Return temperature': 70,     'n': 1.3},
        (1961, 1972):       {'Supply temperature': 80,     'Return temperature': 60,     'n': 1.3},
        (1973, 1978):       {'Supply temperature': 80,     'Return temperature': 40,     'n': 1.3},
        (1979, 1998):       {'Supply temperature': 70,     'Return temperature': 40,     'n': 1.3},
        (1999, 2012):       {'Supply temperature': 70,     'Return temperature': 40,     'n': 1.3},
        (2013, now_yr):     {'Supply temperature': 60,     'Return temperature': 40,     'n': 1.3},
    }

    # Run and save

    for label in building_XML:
        xmls = label["xml"]
            
        for xml in xmls:    
            if xml is None:
                continue
            
            # GET TYPICAL SUPPLY AND RETURN TEMPERATURES FOR THE CONSTRUCTION YEAR
            for years, values in typical_supplyreturn.items():
                start, end = years
                if start <= xml['BBR']['YearOfConstruction'] <= end:
                    xml["DesignCondition"]["T design supply"] = values['Supply temperature']
                    xml["DesignCondition"]["expected n"] = values['n']
                    
                    if (not xml['heatingsystem'].empty) and any(xml['heatingsystem']['Type of system'] == 'Unified'):
                        xml["DesignCondition"]["T design return"] = values['Supply temperature'] - 10
                    else:
                        xml["DesignCondition"]["T design return"] = values['Return temperature']
                    
                    break
            
            values = calc_design_values(xml, getDesignRange = "Min", rad_n = copy.deepcopy(n))
            xml["DesignCondition"]["Design Heatloss Min"]["Radiator parameter"] = values[0]
            xml["DesignCondition"]["Design Heatloss Min"]["Design massFlow"] = values[1]
            
            values = calc_design_values(xml, getDesignRange = "Max", rad_n = copy.deepcopy(n))
            xml["DesignCondition"]["Design Heatloss Max"]["Radiator parameter"] = values[0]
            xml["DesignCondition"]["Design Heatloss Max"]["Design massFlow"] = values[1]
            
            xml["DesignCondition"]["Design Heatloss Current"]["Temperatures - Design Min"] = calc_temperature_values(xml, getDesignRange = "Min")
            xml["DesignCondition"]["Design Heatloss Current"]["Temperatures - Design Max"] = calc_temperature_values(xml, getDesignRange = "Max")
    
    with open(path_save_data, 'wb') as f:
        pickle.dump(building_XML, f)     

