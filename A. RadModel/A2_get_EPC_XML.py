# -*- coding: utf-8 -*-
"""
A2_get_EPC_XML.py

Retrieves the full EPC (Energy Performance Certificate) XML report for each EPC
Serial Identifier found in A1_get_EPC_Identifier.py, via the EMOWeb API.

Extracts all EPCs, including expired ones, and parses out the building
characteristics (BBR data, zones, building parts, windows, heating/hot water
systems, ventilation and internal heat gains) into per-building dictionaries
of dataframes.

Output: a pickle file with the original EPC identifier records, each enriched
with the parsed EPC class label and the parsed building XML data.
"""

import requests
from requests.auth import HTTPBasicAuth
import pickle
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict, Any


def get_XML(EnergyLabelSerialIdentifier: str, EMAIL: str, PASSWORD: str) -> requests.Response:
    """
    Fetch the raw EPC XML report for a given EPC Serial Identifier.

    EnergyLabelSerialIdentifier: str
        The EPC Serial Identifier to fetch the XML report for.
    EMAIL: str
        EMOWeb account email used for HTTP basic authentication.
    PASSWORD: str
        EMOWeb account password used for HTTP basic authentication.

    Returns
    -------
    requests.Response
        The raw HTTP response containing the EPC XML report.
    """
    # CONSTRUCT URL USING SERIAL IDENTIFIER - we use FullRaw as it contains more information in the PDF part of the XML than the Raw
    URL = 'https://emoweb.dk/EMOData/EMOData.svc/FetchEnergyLabelXmlFullRaw/{}/TL,BA'.format(EnergyLabelSerialIdentifier)

    # FETCH XML DATA FROM URL
    response = requests.get(URL, auth=HTTPBasicAuth(EMAIL, PASSWORD))

    return response

def extract_BBR_data(building_xml):

    building_data = {}
    
    for BBR in building_xml.findall(".//BBR"):
        
        building_data['BuildingNumber'] = int(BBR.find("BuildingNumber").text) if BBR is not None else None
        building_data['UseCode'] = int(BBR.find("UseCode").text) if BBR is not None else None
        building_data['YearOfConstruction'] = int(BBR.find("YearOfConstruction").text) if BBR is not None else None
        
        try: 
            building_data['YearOfRenovation'] = int(BBR.find("YearOfRenovation").text) if BBR is not None else None
        except:
            building_data['YearOfRenovation'] = None if BBR is not None else None
            
        building_data['DwellingArea'] = float(BBR.find("DwellingArea").text) if BBR is not None else None
        building_data['CommercialArea'] = float(BBR.find("CommercialArea").text) if BBR is not None else None
    
    return building_data

def extract_EPCLabel_data(EPC_identifier: str, EMAIL: str, PASSWORD: str):
    """
    Retrieve and parse the overall EPC energy label classification (e.g. "A2020", "C")
    for a given EPC Serial Identifier.

    EPC_identifier: str
        The EPC Serial Identifier to fetch the label for.
    EMAIL: str
        EMOWeb account email used for HTTP basic authentication.
    PASSWORD: str
        EMOWeb account password used for HTTP basic authentication.

    Returns
    -------
    str or None
        The EPC energy label classification, or None if it could not be retrieved.
    """
    try: 
        # GET EMOWEB DATA FROM FUNCTION
        xml = get_XML(EPC_identifier, EMAIL, PASSWORD)
    
        # SAVE XML AS FILE
        with open('tmp_emoweb.xml', 'wb') as file:
            file.write(xml.content)
    
        # PARSE THE XML FILE
        tree = ET.parse('tmp_emoweb.xml')
        root = tree.getroot()
    
        # ONLY SAVE BUILDING DATA FROM INPUTDATA
        results_xml = root.findall('.//{http://www.ens.dk/ResultData}ResultData')
    
        for result_xml in results_xml:
            # REMOVE ALL NAMESPACES
            for item in result_xml.iter():
                if '}' in item.tag:
                    item.tag = item.tag.split('}', 1)[1]
        
            # CREATE TREE
            result_tree = ET.ElementTree(result_xml)
        
            # SAVE TREE AS XML
            result_tree.write('tmp_result.xml', encoding='utf-8', xml_declaration=True)        
        
            for result in result_xml.findall(".//LabelResults"):
                
                try:
                    label = result.find("EnergyLabelClassification").text if result is not None else None
                    #print(label)
                except:
                    for result in result.findall(".//Status"):
                        label = result.find("EnergyLabelClassification").text if result is not None else None
                        #print(label)
    except:
        print('INFO: No XML was fetched for the provided EPC Identifier')
        label = None
    
    return label

def extract_zone_data(building_xml):
    
    zones = []
    
    for zone in building_xml.findall(".//Zone"):
        
        if len(zone) != 0:
            zone_data = {}
            
            zone_data['ZoneName'] = zone.find("Name").text if zone is not None else None
            
            try: 
                zone_data['TotalHeatedFloorArea'] = float(zone.find("TotalHeatedFloorArea").text) if zone is not None else None
            except:
                zone_data['TotalHeatedFloorArea'] = float(zone.find("HeatedDwellingArea").text) + float(zone.find("HeatedCommercialArea").text)  if zone is not None else None
            
            zone_data['HeatCapacity'] = float(zone.find("HeatCapacity").text) if zone is not None else None
            zone_data['SetPointHeatedTemperature'] = float(zone.find("SetPointHeatedTemperature").text) if zone is not None else None
            zone_data['SetPointWantedTemperature'] = float(zone.find("SetPointWantedTemperature").text) if zone is not None else None
            zone_data['UsageDays'] = float(zone.find("UsageDays").text) if zone is not None else None
            zone_data['UsageFrom'] = float(zone.find("UsageFrom").text) if zone is not None else None
            zone_data['UsageTo'] = float(zone.find("UsageTo").text) if zone is not None else None
                         
        zones.append(zone_data)
    
    return pd.DataFrame(zones)

def get_selected_elements(status,elementName):
    
    shorttext = status.find("ShortText").text
    
    try: 
        classification = status.find("BIClassification").text
    except:
        classification = status.find("SEEBClassification").text
    
    parts = status.findall(".//BuildingUnit/{}".format(elementName)) 
    if len(parts) == 0:
        parts = status.findall(".//BuildingUnit")
        parts = [part for part in parts if part.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}type', '').endswith(':{}'.format(elementName))]

    # Filters out the parts descriptions belonging to proposals. 
    proposals = status.find("Proposals")
    
    if proposals is not None:

        parts_proposals = proposals.findall(".//BuildingUnit/{}".format(elementName)) 
        if len(parts_proposals) == 0:
            parts_proposals = proposals.findall(".//BuildingUnit")
            parts_proposals = [part for part in parts_proposals if part.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}type', '').endswith(':{}'.format(elementName))]
        
        parts = [item for item in parts if item not in parts_proposals]

    return parts, shorttext, classification

def extract_buildingpart_data(building_xml):

    buildingparts_data = []

    for status in building_xml.findall(".//Status"):
        
        buildingparts, shorttext, classification = get_selected_elements(status, "BuildingPart")
        
        for buildingpart in buildingparts:
            
            if len(buildingpart) != 0: 
                buildingpart_data = {}
                
                buildingpart_data['Description'] = shorttext if buildingpart is not None else None
                buildingpart_data['Classification'] = classification if buildingpart is not None else None
                buildingpart_data['Area'] = float(buildingpart.find("Area").text) if buildingpart is not None else None
                buildingpart_data['Uvalue'] = float(buildingpart.find("UValue").text) if buildingpart is not None else None
                buildingpart_data['Bfactor'] = float(buildingpart.find("BFactor").text) if buildingpart is not None else None
                buildingpart_data['Dimensioning outer temperature'] = float(buildingpart.find("DimensioningOuterTemperature").text) if buildingpart is not None else None
                buildingpart_data['Dimensioning inner temperature'] = float(buildingpart.find("DimensioningInnerTemperature").text) if buildingpart is not None else None
                
            buildingparts_data.append(buildingpart_data)
        
    return pd.DataFrame(buildingparts_data)

def extract_window_data(building_xml):
    
    windows_data = []

    for status in building_xml.findall(".//Status"):
        
        windows, shorttext, classification = get_selected_elements(status, "Window")

        for window in windows:
            
            if len(window) != 0:
                window_data = {}
                
                window_data['Description'] = shorttext if window is not None else None
                window_data['Classification'] = classification if window is not None else None
                window_data['Number'] = float(window.find("NumberOfWindows").text) if window is not None else None
                window_data['Orientation'] = float(window.find("Orientation").text) if window is not None else None
                window_data['Inclination'] = float(window.find("Inclination").text) if window is not None else None
                window_data['Area'] = float(window.find("Area").text) if window is not None else None
                window_data['Uvalue'] = float(window.find("UValue").text) if window is not None else None
                window_data['Bfactor'] = float(window.find("BFactor").text) if window is not None else None
                window_data['Glassshare'] = float(window.find("GlassShare").text) if window is not None else None
                window_data["Solar heat transmittance"] = float(window.find("SolarHeatTransmittance").text) if window is not None else None
                window_data["Solar screening"] = float(window.find("SolarScreening").text) if window is not None else None

            windows_data.append(window_data)
    
    return pd.DataFrame(windows_data)

def extract_heatingsystem_data(building_xml):
    
    systems_data = []

    for status in building_xml.findall(".//Status"):
        
        systems, shorttext, classification = get_selected_elements(status, "HeatDistributionSystem")

        for system in systems:
            
            if len(system) != 0:
                system_data = {}
                
                system_data['Description'] = shorttext if system is not None else None
                system_data['Classification'] = classification if system is not None else None
                system_data['Supply pipe temperature'] = float(system.find("SupplyPipeTemperature").text) if system is not None else None
                system_data['Return pipe temperature'] = float(system.find("ReturnPipeTemperature").text) if system is not None else None
                system_data['Type of system'] = system.find("TypeOfSystem").text if system is not None else None

            systems_data.append(system_data)
        
    return pd.DataFrame(systems_data)

def extract_hotwatersystem_data(building_xml):
    
    systems_data = []

    for status in building_xml.findall(".//Status"):
        
        systems, shorttext, classification = get_selected_elements(status, "HotWaterTank")

        for system in systems:
            
            if len(system) != 0:
                system_data = {}
            
                system_data['Description'] = shorttext if system is not None else None
                system_data['Classification'] = classification if system is not None else None
                system_data['NumberOfTanks'] = float(system.find("NumberOfTanks").text) if system is not None else None
                system_data['ShareOfHotWaterConsumption'] = float(system.find("ShareOfHotWaterConsumption").text) if system is not None else None
                system_data['Volume'] = float(system.find("Volume").text) if system is not None else None
                system_data['SupplyTemperature'] = float(system.find("SupplyTemperature").text) if system is not None else None
                system_data['ElectricalSupport'] = system.find("ElectricalSupport").text if system is not None else None
                system_data['HasHeatingCoil'] = system.find("HasHeatingCoil").text if system is not None else None
                system_data['HeatLoss'] = float(system.find("HeatLoss").text) if system is not None else None
                system_data['BFactor'] = float(system.find("BFactor").text) if system is not None else None
                
                systems_data.append(system_data)

    return pd.DataFrame(systems_data)

def extract_hotwaterconsumption_data(building_xml):
    
    systems_data = []
    
    for status in building_xml.findall(".//Status"):
        
        systems, shorttext, classification = get_selected_elements(status, "HotWaterConsumption")

        for system in systems:
            
            if len(system) != 0:
                system_data = {}
            
                system_data['Description'] = shorttext if system is not None else None
                system_data['Classification'] = classification if system is not None else None
                system_data['AverageConsumption'] = float(system.find("AverageConsumption").text) if system is not None else None
                system_data['Temperature'] = float(system.find("Temperature").text) if system is not None else None

            systems_data.append(system_data)

    return pd.DataFrame(systems_data)

def extract_districtheatingindirect_data(building_xml):
    
    systems_data = []

    for status in building_xml.findall(".//Status"):
        
        systems, shorttext, classification = get_selected_elements(status, "DistrictHeatWithExchanger")

        for system in systems:
            
            if len(system) != 0:
                system_data = {}
            
                system_data['Description'] = shorttext if system is not None else None
                system_data['Classification'] = classification if system is not None else None
                system_data['DistrictHeatingPlantName'] = system.find("DistrictHeatingPlantName").text if system is not None else None
                system_data['HeatLoss'] = float(system.find("HeatLoss").text) if system is not None else None
                system_data['BFactor'] = float(system.find("BFactor").text) if system is not None else None
                system_data['NominalEffect'] = float(system.find("NominalEffect").text) if system is not None else None
                system_data['MinimumTemperature'] = float(system.find("MinimumTemperature").text) if system is not None else None
                system_data['HeatsHotWater'] = system.find("HeatsHotWater").text if system is not None else None
        
            systems_data.append(system_data)

    return pd.DataFrame(systems_data)

def extract_districtheatingdirect_data(building_xml):
    
    systems_data = []

    for status in building_xml.findall(".//Status"):
        
        systems, shorttext, classification = get_selected_elements(status, "DirectDistrictHeat")

        for system in systems:
            
            if len(system) != 0:
                system_data = {}
            
                system_data['Description'] = shorttext if system is not None else None
                system_data['Classification'] = classification if system is not None else None
                system_data['DistrictHeatingPlantName'] = system.find("DistrictHeatingPlantName").text if system is not None else None
                system_data['IsPrimary'] = system.find("IsPrimary").text if system is not None else None
        
            systems_data.append(system_data)

    return pd.DataFrame(systems_data)

def extract_districtheatingOther_data(building_xml):
    
    systems_data = []

    for status in building_xml.findall(".//Status"):
        
        systems, shorttext, classification = get_selected_elements(status, "DistrictHeat")

        for system in systems:
            
            if len(system) != 0:
                system_data = {}
            
                system_data['Description'] = shorttext if system is not None else None
                system_data['DistrictHeatingPlantName'] = system.find("DistrictHeatingPlantName").text if system is not None else None
                system_data['IsPrimary'] = system.find("IsPrimary").text if system is not None else None
        
            systems_data.append(system_data)

    return pd.DataFrame(systems_data)

def extract_lineloss_data(building_xml):
    
    linelosses_data = []

    for status in building_xml.findall(".//Status"):
        
        linelosses, shorttext, classification = get_selected_elements(status, "LinearThermalTransmittance")

        for lineloss in linelosses:
            
            if len(lineloss) != 0:
                lineloss_data = {}
            
                lineloss_data['Description'] = shorttext if lineloss is not None else None
                lineloss_data['Classification'] = classification if lineloss is not None else None
                lineloss_data['Length'] = float(lineloss.find("Length").text) if lineloss is not None else None
                lineloss_data['Heatloss'] = float(lineloss.find("HeatLoss").text) if lineloss is not None else None
                lineloss_data['Bfactor'] = float(lineloss.find("BFactor").text) if lineloss is not None else None
                lineloss_data['Dimensioning outer temperature'] = float(lineloss.find("DimensioningOuterTemperature").text) if lineloss is not None else None
                lineloss_data['Dimensioning inner temperature'] = float(lineloss.find("DimensioningInnerTemperature").text) if lineloss is not None else None
                
                linelosses_data.append(lineloss_data)
        
    return pd.DataFrame(linelosses_data)

def extract_ventilation_data(building_xml):
    
    ventilations_data = []

    for status in building_xml.findall(".//Status"):
        
        ventilations, shorttext, classification = get_selected_elements(status, "Ventilation")

        for ventilation in ventilations:
            
            if len(ventilation) != 0:
                ventilation_data = {}
            
                ventilation_data['Description'] = shorttext if ventilation is not None else None
                ventilation_data['Classification'] = classification if ventilation is not None else None
                ventilation_data['Area'] = float(ventilation.find("Area").text) if ventilation is not None else None
                ventilation_data['UsageFactor'] = float(ventilation.find("UsageFactor").text) if ventilation is not None else None
                ventilation_data['NaturalVentilationWinter'] = float(ventilation.find("NaturalVentilationWinter").text) if ventilation is not None else None
                ventilation_data['NaturalVentilationSummer'] = float(ventilation.find("NaturalVentilationSummer").text) if ventilation is not None else None
                ventilation_data['NaturalVentilationSummerOutsideUsageHours'] = float(ventilation.find("NaturalVentilationSummerOutsideUsageHours").text) if ventilation is not None else None
                ventilation_data['MechanicalVentilationWinter'] = float(ventilation.find("MechanicalVentilationWinter").text) if ventilation is not None else None
                ventilation_data['MechanicalVentilationSummer'] = float(ventilation.find("MechanicalVentilationSummer").text) if ventilation is not None else None
                ventilation_data['MechanicalVentilationSummerOutsideUsageHours'] = float(ventilation.find("MechanicalVentilationSummerOutsideUsageHours").text) if ventilation is not None else None
                ventilation_data['InfiltrationOutsideUsageHours'] = float(ventilation.find("InfiltrationOutsideUsageHours").text) if ventilation is not None else None
                ventilation_data['TemperatureEfficiency'] = float(ventilation.find("TemperatureEfficiency").text) if ventilation is not None else None
                ventilation_data['InjectionTemperature'] = float(ventilation.find("InjectionTemperature").text) if ventilation is not None else None
                ventilation_data['SpecialPowerUsage'] = float(ventilation.find("SpecialPowerUsage").text) if ventilation is not None else None
                ventilation_data['HasHeatCoils'] = ventilation.find("HasHeatCoils").text if ventilation is not None else None
                
                ventilations_data.append(ventilation_data)
        
    return pd.DataFrame(ventilations_data)

def extract_internalHeatGain_data(building_xml):
    
    internalHeatGains_data = []

    for status in building_xml.findall(".//Status"):
        
        internalHeatGains, shorttext, classification = get_selected_elements(status, "InternalHeatContribution")           

        for internalHeatGain in internalHeatGains:
            
            if len(internalHeatGain) != 0:
                internalHeatGain_data = {}
            
                internalHeatGain_data['Description'] = shorttext if internalHeatGain is not None else None
                internalHeatGain_data['Classification'] = classification if internalHeatGain is not None else None
                internalHeatGain_data['Area'] = float(internalHeatGain.find("Area").text) if internalHeatGain is not None else None
                internalHeatGain_data['PowerFromPersons'] = float(internalHeatGain.find("PowerFromPersons").text) if internalHeatGain is not None else None
                internalHeatGain_data['PowerFromDevicesInUsageHours'] = float(internalHeatGain.find("PowerFromDevicesInUsageHours").text) if internalHeatGain is not None else None
                internalHeatGain_data['PowerFromDevicesOutsideUsageHours'] = float(internalHeatGain.find("PowerFromDevicesOutsideUsageHours").text) if internalHeatGain is not None else None
                
                internalHeatGains_data.append(internalHeatGain_data)
        
    return pd.DataFrame(internalHeatGains_data)

def get_EPC_data(EnergyLabelSerialIdentifier: str, EMAIL: str, PASSWORD: str) -> list:
    """
    Fetch and parse the full EPC XML report for a given EPC Serial Identifier
    into a list of per-building dictionaries of dataframes.

    EnergyLabelSerialIdentifier: str
        The EPC Serial Identifier to fetch and parse.
    EMAIL: str
        EMOWeb account email used for HTTP basic authentication.
    PASSWORD: str
        EMOWeb account password used for HTTP basic authentication.

    Returns
    -------
    list of dict
        One dict per building in the EPC report, containing the parsed BBR,
        zone, building part, window, line loss, heating system, hot water
        system, district heating system, ventilation and internal heat gain
        data. Returns [None] if the XML could not be retrieved or parsed.
    """
    
    buildings = []
        
    try: 
        # GET EMOWEB DATA FROM FUNCTION
        xml = get_XML(EnergyLabelSerialIdentifier, EMAIL, PASSWORD)
    
        # SAVE XML AS FILE
        with open('tmp_emoweb.xml', 'wb') as file:
            file.write(xml.content)
    
        # PARSE THE XML FILE
        tree = ET.parse('tmp_emoweb.xml')
        root = tree.getroot()
    
        # ONLY SAVE BUILDING DATA FROM INPUTDATA
        buildings_xml = root.findall('.//{http://www.ens.dk/InputData}Building')
        
        for building_xml in buildings_xml:
            # REMOVE ALL NAMESPACES
            for item in building_xml.iter():
                if '}' in item.tag:
                    item.tag = item.tag.split('}', 1)[1]
        
            # CREATE TREE
            building_tree = ET.ElementTree(building_xml)
        
            # SAVE TREE AS XML
            building_tree.write('tmp_building.xml', encoding='utf-8', xml_declaration=True)
                
            # FIND BBR DATA IN XML FILE
            BBR = extract_BBR_data(building_xml)
            BBR['Total Area'] = BBR['DwellingArea'] + BBR['CommercialArea']
            
            # Get the DH heating central info
            
            if len(extract_districtheatingindirect_data(building_xml)) != 0:
            
                DH = extract_districtheatingindirect_data(building_xml) 
                
            elif len(extract_districtheatingdirect_data(building_xml)) != 0:
                
                DH = extract_districtheatingdirect_data(building_xml)
                
            else:
                
                DH = extract_districtheatingOther_data(building_xml)
            
            # GET SPECIFIC DATA FROM XML FILE      
            building = {
                "BBR": BBR,
                "zones": extract_zone_data(building_xml),
                "buildingparts": extract_buildingpart_data(building_xml),
                "windows": extract_window_data(building_xml),
                "linelosses": extract_lineloss_data(building_xml),
                "heatingsystem": extract_heatingsystem_data(building_xml),
                "hotwatersystem": extract_hotwatersystem_data(building_xml),
                "hotwaterconsumption": extract_hotwaterconsumption_data(building_xml),
                "districtheatingsystem": DH,
                "ventilation": extract_ventilation_data(building_xml),
                "internalHeatGain": extract_internalHeatGain_data(building_xml)
                }
            
            buildings.append(building)
        
    except:
        print('INFO: No XML was fetched for the provided EPC Identifier')
        buildings.append(None)
    
    return buildings

def retrieve_EPC_XML(path_epc_id: str, path_epc_xml: str, EMAIL: str, PASSWORD: str) -> None:
    """
    Load the EPC identifiers produced by A1_get_EPC_Identifier.py, fetch the
    full EPC XML report for each unique EPC Serial, and save the enriched
    records to `path_epc_xml`.

    path_epc_id: str
        File path (.pkl) of the EPC identifier records produced by
        retrieve_EPC() in A1_get_EPC_Identifier.py.
    path_epc_xml: str
        File path (.pkl) where the enriched records (including the parsed
        EPC class label and parsed building XML data) are saved.
    EMAIL: str
        EMOWeb account email used for HTTP basic authentication.
    PASSWORD: str
        EMOWeb account password used for HTTP basic authentication.

    Returns
    -------
    None
        The result is written to `path_epc_xml` as a pickle file.
    """
    
    with open(path_epc_id, 'rb') as f:
        epc_identifiers = pickle.load(f)
            
    # preserve order of first occurrence
    seen = set()
    unique_serials = []
    for d in epc_identifiers:
        s = d['EPCSerial']
        if s not in seen:
            seen.add(s)
            unique_serials.append(s)

    # fetch once per unique serial
    serial_to_result = {}
    for serial in unique_serials:
        epc_class = extract_EPCLabel_data(serial, EMAIL, PASSWORD)
        xml = get_EPC_data(serial, EMAIL, PASSWORD)
        serial_to_result[serial] = {
            'EPC_Class_Property': epc_class,
            'xml': xml
        }

    # map back to original items so linking is preserved
    buildings_xml = []
    for data in epc_identifiers:
        buildings_xml.append({**data, **serial_to_result[data['EPCSerial']]})
        
    with open(path_epc_xml, 'wb') as f:
        pickle.dump(buildings_xml, f)


