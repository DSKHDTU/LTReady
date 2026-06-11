# -*- coding: utf-8 -*-
"""
A4_get_heatloss.py

Calculates the design heat loss (with and without internal heat gain) for each
building, using the U-value ranges assigned in A3_get_design_values.py.

Dimensioning indoor (Tin) and outdoor (Tout) design temperatures are determined
based on the building's construction year, following historical Danish design
practice (see DS418 and historical publications referenced in A3).

Buildings constructed before BR1961 (<1961) have a design heat loss accounting
only for natural ventilation, since mechanical ventilation was not yet required.
"""

import pickle
from datetime import datetime

def get_heatloss(path_load_data: str, path_save_data: str) -> None:
    """
    Calculate the design heat loss (minimum, maximum and current) for each
    building in the EPC dataset.

    path_load_data: str
        File path (.pkl) of the EPC XML dataset enriched with design U-values
        (output of A3_get_design_values.py).
    path_save_data: str
        File path (.pkl) where the enriched dataset is saved. Can be the same
        as path_load_data to overwrite in place.

    Returns
    -------
    None
        The result is written to `path_save_data` as a pickle file. Adds a
        "DesignCondition" entry to each building's XML dictionary, containing
        the design temperatures and the calculated heat loss components
        ("Design Heatloss Min", "Design Heatloss Max", "Design Heatloss Current").
    """
    
    
    # Upload stamdata
    with open(path_load_data, 'rb') as f:
        building_XML = pickle.load(f)
        
    del f
    
    # Assumptions about historical values
    now_yr = datetime.now().year
    
    # Natural ventilation was the only ventilation form until 1961 (BR-1961)
    mechvent_req_year = 1961
    
    # DS418 Tin and Tout design parameters
    Ti = 20
    To = -12
    
    # Design Tin and Tout
    typical_designTemperature = {
        (1700, 1954):       {'Tin': Ti,     'Tout': {'min': -6, 'max': -15}},
        (1955, 1967):       {'Tin': Ti,     'Tout': {'min': -15, 'max': -15}},
        (1968, now_yr):     {'Tin': Ti,     'Tout': {'min': To, 'max': To}}
        }
    
    # Functions 
    def heatloss_calculation(xml,Uvalue_column_name,Tin,Tout,qa=None,c_air = 1005,rho_air = 1.21,InclMechVent=True,InternalGainsIncluded=False):
        heatloss = {
            'windows': [],
            'buildingparts': [],
            'linelosses': [],
            'ventilation': [],
            'TotalHeatloss': [],
            'InternalHeatGain': [],
            'HeatlossInclGain': []
            }
            
        # Note that the b-factor is calculated for dimensioning temperatures. 
        # If outdoor and indoor temperature changes or heat transfer coefficients change, 
        # the b-factor will also change. Since most of surfaces are facing the outdoor 
        # we assume the b-factor to be the same as design conditions.
        
        windows = xml["windows"]
        buildingparts = xml["buildingparts"]
        linelosses = xml["linelosses"]
        ventilation = xml["ventilation"]
        heated_floor_area = xml["zones"]["TotalHeatedFloorArea"].sum()
                
        # Calculate heat loss
        heatloss['windows'] = (windows["Number"]*windows["Area"]*windows[Uvalue_column_name]*windows["Bfactor"]).sum()*(Tin - Tout) if len(windows) !=0 else 0
        heatloss['buildingparts'] = (buildingparts["Area"]*buildingparts[Uvalue_column_name]*buildingparts["Bfactor"]).sum()*(Tin - Tout) if len(buildingparts) !=0 else 0
        heatloss['linelosses'] = (linelosses["Length"]*linelosses["Heatloss"]*linelosses["Bfactor"]).sum()*(Tin - Tout) if len(linelosses) !=0 else 0
        
        # Ventilation heat loss
        
        ## Natural ventilation
        def calc_nat_vent(x=qa):
            
            if x is None:
                # qa = 0.3 l/s per m2 according to HBE and DS418. m2 is the heated floor area.
                
                if len(ventilation)!= 0:
                    q = (ventilation["Area"]*(ventilation["UsageFactor"]*ventilation["NaturalVentilationWinter"]+(1-ventilation["UsageFactor"])*ventilation["InfiltrationOutsideUsageHours"])).sum()
                else: 
                    q = 0.3*heated_floor_area
                    
            else:
                q = x*heated_floor_area
                
            natural = q*c_air*rho_air/1000*(Tin - Tout)
            
            return natural
        
        ## Mechanical ventilation / accounting for heat recovery / inlet temperature
        def calc_mech_vent(row):
            
            if len(ventilation)!= 0:
                Tinlet = Tout + row["TemperatureEfficiency"]*(Tin - Tout) 
                mechanical = row["Area"]*row["UsageFactor"]*row["MechanicalVentilationWinter"]*c_air*rho_air/1000*(Tin - Tinlet)
            else:
                mechanical = 0
            
            return mechanical
        
        mechanical = ventilation.apply(calc_mech_vent, axis=1).sum() if InclMechVent else 0
        natural = calc_nat_vent()
        
        heatloss['ventilation'] = natural + mechanical   
        
        # Total heat loss
        heatloss['TotalHeatloss'] = heatloss['windows'] + heatloss['buildingparts'] + heatloss['linelosses'] + heatloss['ventilation']
        
        # Heat loss and internal gain
        if InternalGainsIncluded:
            internalgain = xml["internalHeatGain"]
            zone = xml["zones"]
            
            # Occupancy period calculated. A week has 24*7 = 168 hours in total.
            if zone["UsageDays"] > 0 and zone["UsageTo"] == 0:
                Usagefactor = zone["UsageDays"] / 168
            else:
                Usagefactor = (zone["UsageTo"]-zone["UsageFrom"])*zone["UsageDays"] / (24*7)
            
            heatloss['InternalHeatGain'] = internalgain["Area"]*(internalgain["PowerFromPersons"] + internalgain["PowerFromDevicesInUsageHours"]*Usagefactor + internalgain["PowerFromDevicesOutsideUsageHours"]*(1-Usagefactor))
            heatloss['HeatlossInclGain'] = heatloss['TotalHeatloss'] - heatloss['InternalHeatGain'] if heatloss['InternalHeatGain'] < heatloss['TotalHeatloss'] else 0
    
        else:
            heatloss['InternalHeatGain'] = None
            heatloss['HeatlossInclGain'] = None
            
        return heatloss
    
    #
    for label in building_XML:
        
        xmls = label["xml"]
        
        for xml in xmls:    
            if xml is None:
                continue
        
            xml["DesignCondition"] = {
                "Tin design":[],
                "Design Heatloss Min": {"Tout design": []},
                "Design Heatloss Max": {"Tout design": []}
                }
            
            # PLACE IN THE CORRECT YEAR RANGE
            for years, values in typical_designTemperature.items():
                start, end = years
                
                if start <= xml["BBR"]["YearOfConstruction"] <= end:
                    xml["DesignCondition"]["Tin design"] = values['Tin']
                    xml["DesignCondition"]["Design Heatloss Min"]["Tout design"] = values['Tout']['min']
                    xml["DesignCondition"]["Design Heatloss Max"]["Tout design"] = values['Tout']['max']
                
                if xml["BBR"]["YearOfConstruction"] < mechvent_req_year:
                    xml["DesignCondition"]["InclMechanicalVentilationLoss"] = False
                    
                else:
                    xml["DesignCondition"]["InclMechanicalVentilationLoss"] = True
            
            # In Regler for varmetabsberegning, 1953 1. udg the minimum value for accounting for infiltration 12 Wh/m2 (this matches with 0.3/1000*35*1.2*1000 = 13 Wh/m2)
            # So the infiltration we 
            
            xml["DesignCondition"]["Design Heatloss Min"].update(
                heatloss_calculation(xml,
                                     Uvalue_column_name = "Design Uvalue Min", 
                                     Tin = xml["DesignCondition"]["Tin design"], 
                                     Tout = xml["DesignCondition"]["Design Heatloss Min"]["Tout design"],
                                     InclMechVent = xml["DesignCondition"]["InclMechanicalVentilationLoss"],
                                     qa = None if xml["DesignCondition"]["InclMechanicalVentilationLoss"] else 0.3)
                )
            
            xml["DesignCondition"]["Design Heatloss Max"].update(
                heatloss_calculation(xml,
                                     Uvalue_column_name = "Design Uvalue Max", 
                                     Tin = xml["DesignCondition"]["Tin design"], 
                                     Tout = xml["DesignCondition"]["Design Heatloss Max"]["Tout design"],
                                     InclMechVent = xml["DesignCondition"]["InclMechanicalVentilationLoss"],
                                     qa = None if xml["DesignCondition"]["InclMechanicalVentilationLoss"] else 0.3)
                )
            
            xml["DesignCondition"]["Design Heatloss Current"] = heatloss_calculation(xml,
                                                                                     "Uvalue", 
                                                                                     Ti, 
                                                                                     To,
                                                                                     InclMechVent=True,
                                                                                     qa = None)        
    
    # Save XML
    
    with open(path_save_data, 'wb') as f:
        pickle.dump(building_XML, f) 


