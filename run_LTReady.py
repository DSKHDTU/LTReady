# -*- coding: utf-8 -*-
"""
run_LTReady.py

Main script for the LTReady tool. Adjust the USER-SPECIFIC INPUT section
below, then run this script to execute the full workflow.

The workflow is divided into three parts:

    Part A: Obtains the EPC (Energy Performance Certificate) dataset and the
            design radiator nominal/design values for each building.
    Part B: Obtains the predicted peak space heating (SH) consumption for
            each meter/component, based on a data-driven heat loss model.
    Part C: Calculates the minimum supply and return temperatures achievable
            for each component, and produces the final dataset and map for
            analysis.

Required input data
--------------------
1. Master ID file (`master_ids_{CASE_NAME}.pkl`): a pickled pandas DataFrame with one row
   per property, containing at least the columns:
       - "BFEnummer"    : BFE number (Danish property identifier), or NaN (can be obtained via https://datafordeler.dk/)
       - "husnummer_ID" : address unit ID, or NaN (can be obtained via https://datafordeler.dk/)
       - "vejnavn", "husnummer" : street name / house number (fallback display
         fields if no EPC is found, optional)

2. Component ID file (`component_ids_{CASE_NAME}.pkl`): a pickled pandas DataFrame defining how
   buildings/meters are grouped into "components" (e.g. all buildings sharing
   a single district heating substation), containing at least the columns:
       - "component_id" : user-defined ID for each component
       - "bfe"          : BFE number(s) belonging to this component (scalar,
                           list, or stringified list)
       - "epc"          : EPC Serial(s) belonging to this component (optional,
                           same format as "bfe")
       - "serial"       : meter serial number(s) belonging to this component
                           (scalar, list, or stringified list)
       - "location_id"  : meter location number(s) belonging to this component
                           (scalar, list, or stringified list)
       - "n_serial"     : number of meter serials in this component

3. Raw hourly meter data (`hourly_meter_raw_{CASE_NAME}_{YEAR1}`, `hourly_meter_raw_{CASE_NAME}_{YEAR2}`): parquet
   files with hourly heat meter readings, containing at least "timestamp",
   "serial","location_id", "energy", "volume", "heat_power", "volume_flow", "inlet_flow_temperature",
   "return_flow_temperature", and any columns listed in `meter_id_list`.
   Raw hourly meter data MUST contain a data for a whole year, incl. months in June, July or August for correct disaggregation of SH consumption from total consumption.
   serial and location_id preferably as strings.
   values for each measurement variable must be: 
      - energy - Wh
      - heat_power - W
      - volume - m3
      - volume_flow - m3/s
      - inlet_flow_temperature - °C
      - return_flow_temperature - °C

The filenames of the three input datasets must match as specified.
The column/variable names of the datasets must match the variable names listed 
above.

@author: dskh
"""

#%% OVERALL LIBRARIES
import pickle
import pandas as pd
import os
import itertools
import sys

#%% USER-SPECIFIC INPUT

# A short identifier for this case study, used as a suffix in all generated
# file names.
CASE_NAME = "EXAMPLE_CITY_NAME" #

# Calendar year(s) of meter data available. If only one year of meter data is
# available, set YEAR2 = None and HD_year2 = None; the workflow will then skip
# all year-2-specific and cross-year processing steps.
YEAR1 = 2022
YEAR2 = 2023  # Set to None if only one year of meter data is available

# ---------------------------------------------------------------------- #
# EMOWeb credentials (Part A) - used to retrieve EPC data via the EMOWeb API
# ---------------------------------------------------------------------- #
EMAIL = "email@email.com" #
PASSWORD = "PASSWORD" #

# ---------------------------------------------------------------------- #
# DMI Open Data API key (Part B) - used to retrieve weather data
# DMI weather station IDs used to retrieve each weather parameter. These can
# all be the same station, or different stations, depending on which nearby
# stations measure each parameter. Find station IDs at
# https://www.dmi.dk/friedata/observationer/
# ---------------------------------------------------------------------- #
STATION_TEMPERATURE = "STATION_ID_INCL_TEMPERATURE" #
STATION_WIND = "STATION_ID_INCL_WIND_DIRECTION_AND_WIND_SPEED" #
STATION_SOLAR_RADIATION = "STATION_ID_INCL_SUN_RADIATION" #

# ---------------------------------------------------------------------- #
# Meter and component identifiers
# ---------------------------------------------------------------------- #

# Column(s) used to uniquely identify a meter. Used for grouping/merging
# throughout Part B.
meter_id_list = ['serial']

# ---------------------------------------------------------------------- #
# Outdoor design conditions for the predicted heat loss / supply-return
# temperature calculations (Part B and C)
# ---------------------------------------------------------------------- #

# Outdoor design temperatures [degC] to evaluate
Tout = [-15, -12, -7.6, -6, 0, 5, 10, 15]
Tout_peak = -12

# Outdoor design wind speeds [m/s] to evaluate
Ws = [0, 2.5, 5, 7.5, 10]
Ws_peak = 5

# Indoor design temperature [degC]. [20] is the recommended value.
Tin = [20]
Tin_peak = 20

# Radiator exponent. 1.3 is the recommended value.
Radn = 1.3

# Radiator sizing factor - select the index for which fsb to plot
fs = [0.85,1.0,1.05]
ifs = 2

# Which model parameter statistic(s) to use: central estimate ("Coef") and/or
# the 95% confidence bounds ("Conf_up", "Conf_low").
paramb = ['Coef', 'Conf_up', 'Conf_low']

# ---------------------------------------------------------------------- #
# Heating degree-days (Part B)
# Used to normalize the predicted annual energy consumption to a reference
# (long-term average) weather year. Obtain the observed heating degree-days
# for your location and year(s) of interest from your national meteorological
# institute.
# ---------------------------------------------------------------------- #
HD_year1 = 2547.3   # Heating degree-days for YEAR1
HD_year2 = 2585.1   # Heating degree-days for YEAR2 (set to None if YEAR2 is None)

# ---------------------------------------------------------------------- #
# Outlier detection (Part C)
# Components whose predicted annual SH energy use (kWh/m2) falls below
# `lower_limit` or above `upper_limit` are flagged as outliers. These are
# quantiles between 0 and 1 (e.g. 0.05 = 5th percentile), computed
# separately for buildings constructed before and from 1960 onwards.
# Set lower_limit=0 and upper_limit=1 to disable this kWh/m2-based check
# (components can still be flagged as outliers based on building use code
# or absence of district heating - see find_outlier_energy()).
# ---------------------------------------------------------------------- #
lower_limit = 0
upper_limit = 1

# ---------------------------------------------------------------------- #
# Heat exchanger temperature offset (Part C)
# For indirectly connected district heating (DH) substations: specify a value
# to account for the heat exchanger efficiency between the primary and
# secondary side. This value is added to the calculated supply and return
# temperatures to enable comparison to primary-side / network temperatures.
# For secondary-side temperatures, or directly connected DH substations, set
# this value to 0.
# ---------------------------------------------------------------------- #
temp_HEX = 5

# ---------------------------------------------------------------------- #
# Final dataset / plotting selection (Part C)
# The supply/return temperatures are calculated under building-level hydraulic 
# constraints (BC) or network-level hydraulic constraints (NC)
# BC uses the calculated design mass flow. Network-level constraint fixes
# temperature drop to 10 or 20 C at peak condition.
# ---------------------------------------------------------------------- #

# Toggle to select the final dataset / plot based on the "BC" method
# (True) or the "NC" method (False). See C1_get_actualDHTemperatures.py
# for a description of both methods.
get_BC = True

# Variables available for analysis/plotting using the "BC" method
var_BC = ['Minimum_Treturn',  # Minimum return temperature using model-predicted heat loss as current peak heat loss
             'Minimum_Tsupply',  # Minimum supply temperature using model-predicted heat loss as current peak heat loss
             'Historical_Nominal_Massflow', # Nominal mass flow
             'Radiator_K',               # Nominal K-value for the radiator system
             'Historical_Nominal_Totalheatloss',   # Nominal heat loss
             'Historical_Nominal_Tout',           # Nominal outdoor design temperature
             'EPC_calculated_Treturn',            # Minimum return temperature using EPC-calculated heat loss as current peak heat loss
             'EPC_calculated_Tsupply']            # Minimum supply temperature using EPC-calculated heat loss as current peak heat loss

# Variables available for analysis/plotting using the "NC" method
var_NC = var_BC + ['Current_Nominal_Totalheatloss', # Peak heat loss saved to verify the peak heat loss is preserved by the NC method
                          'Current_Nominal_Massflow']      # Peak mass flow at the chosen temperature drop (10 or 20 degC)
                     
# Index into var_BC / var_NC selecting the variable to obtain final dataset (starting from 0. var_BC ends with index 7.)
ind = 1

# ---------------------------------------------------------------------- #
# Model reproducibility (Part B)
# Set a fixed integer seed for the train/test split in B6 to ensure
# reproducible model parameters across runs. Set to None to disable.
# ---------------------------------------------------------------------- #
RANDOM_STATE = 42

#%% CONFIGURATION OF FOLDERS
# Assumes the script is run from the project root
PROJECT_ROOT = os.getcwd()  

for subfolder in ["A. RadModel", "B. SHModel", "C. Mapping"]:
    sys.path.append(os.path.join(PROJECT_ROOT, subfolder))

# Define directory paths
DIR_DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DIR_DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DIR_DATA_FINAL = os.path.join(PROJECT_ROOT, "data", "final")

# Ensure the directories exist
os.makedirs(DIR_DATA_RAW, exist_ok=True)
os.makedirs(DIR_DATA_PROCESSED, exist_ok=True)
os.makedirs(DIR_DATA_FINAL, exist_ok=True)

# ---------------------------------------------------------------------- #
# INPUT FILES
# ---------------------------------------------------------------------- #

path_comp_id = os.path.join(DIR_DATA_RAW, f"component_ids_{CASE_NAME}.pkl")
path_master_id = os.path.join(DIR_DATA_RAW, f"master_ids_{CASE_NAME}.pkl")
path_meter_year1 = os.path.join(DIR_DATA_RAW, f"hourly_meter_raw_{CASE_NAME}_{YEAR1}")
path_meter_year2 = os.path.join(DIR_DATA_RAW, f"hourly_meter_raw_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None

# ---------------------------------------------------------------------- #
# Load datasets used across the workflows
# ---------------------------------------------------------------------- #

with open(path_master_id, 'rb') as f:
    master_ids = pickle.load(f)
    
with open(path_comp_id, 'rb') as f:
    compID = pickle.load(f)

# ---------------------------------------------------------------------- #
# PROCESS FILES
# ---------------------------------------------------------------------- #

# 1st workflow (Part A):
path_epc_id = os.path.join(DIR_DATA_PROCESSED, f"unique_epc_id_{CASE_NAME}.pkl")
path_epc_xml = os.path.join(DIR_DATA_PROCESSED, f"epc_xml_{CASE_NAME}.pkl")
path_bbr_xml = os.path.join(DIR_DATA_PROCESSED, f"bbr_xml_{CASE_NAME}.pkl")
path_current_design = os.path.join(DIR_DATA_PROCESSED, f"current_design_xml_{CASE_NAME}.pkl")
path_max_design = os.path.join(DIR_DATA_PROCESSED, f"max_org_design_xml_{CASE_NAME}.pkl")
path_min_design = os.path.join(DIR_DATA_PROCESSED, f"min_org_design_xml_{CASE_NAME}.pkl")

# 2nd workflow (Part B):
path_proc_meter_year1 = os.path.join(DIR_DATA_PROCESSED, f"hourly_meter_proc_{CASE_NAME}_{YEAR1}")
path_proc_meter_year2 = os.path.join(DIR_DATA_PROCESSED, f"hourly_meter_proc_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None
path_SH_year1 = os.path.join(DIR_DATA_PROCESSED, f"hourly_SH_{CASE_NAME}_{YEAR1}")
path_SH_year2 = os.path.join(DIR_DATA_PROCESSED, f"hourly_SH_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None
path_proc_SH_year1 = os.path.join(DIR_DATA_PROCESSED, f"hourly_SH_proc_{CASE_NAME}_{YEAR1}")
path_proc_SH_year2 = os.path.join(DIR_DATA_PROCESSED, f"hourly_SH_proc_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None
path_timeseries = os.path.join(DIR_DATA_PROCESSED, f"timeseries_{CASE_NAME}")
path_model_params = os.path.join(DIR_DATA_PROCESSED, f"model_params_{CASE_NAME}")
path_pred_heatloss = os.path.join(DIR_DATA_PROCESSED, f"pred_heat_loss_{CASE_NAME}")
path_annual_SH_year1 = os.path.join(DIR_DATA_PROCESSED, f"annual_SH_pred_{CASE_NAME}_{YEAR1}")
path_annual_SH_year2 = os.path.join(DIR_DATA_PROCESSED, f"annual_SH_pred_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None

# ---------------------------------------------------------------------- #
# FINAL FILES (3rd workflow, Part C)
# ---------------------------------------------------------------------- #
path_weather_year1 = os.path.join(DIR_DATA_FINAL, f"hourly_weather_{CASE_NAME}_{YEAR1}")
path_weather_year2 = os.path.join(DIR_DATA_FINAL, f"hourly_weather_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None

path_agg_bbr = os.path.join(DIR_DATA_FINAL, f"agg_bbr_xml_{CASE_NAME}.pkl")
path_agg_current = os.path.join(DIR_DATA_FINAL, f"agg_current_design_{CASE_NAME}.pkl")
path_agg_max = os.path.join(DIR_DATA_FINAL, f"agg_max_org_design_{CASE_NAME}.pkl")
path_agg_min = os.path.join(DIR_DATA_FINAL, f"agg_min_org_design_{CASE_NAME}.pkl")

path_agg_proc_SH_year1 = os.path.join(DIR_DATA_FINAL, f"hourly_agg_SH_proc_{CASE_NAME}_{YEAR1}")
path_agg_proc_SH_year2 = os.path.join(DIR_DATA_FINAL, f"hourly_agg_SH_proc_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None
path_agg_pred_heatloss = os.path.join(DIR_DATA_FINAL, f"pred_agg_heat_loss_{CASE_NAME}")
path_agg_annual_SH_year1 = os.path.join(DIR_DATA_FINAL, f"annual_agg_SH_pred_{CASE_NAME}_{YEAR1}")
path_agg_annual_SH_year2 = os.path.join(DIR_DATA_FINAL, f"annual_agg_SH_pred_{CASE_NAME}_{YEAR2}") if YEAR2 is not None else None

path_agg_max_with_SH_BC = os.path.join(DIR_DATA_FINAL, f"agg_max_org_design_with_SH_BC_{CASE_NAME}.pkl")
path_agg_min_with_SH_BC = os.path.join(DIR_DATA_FINAL, f"agg_min_org_design_with_SH_BC_{CASE_NAME}.pkl")
path_agg_max_with_SH_NC = os.path.join(DIR_DATA_FINAL, f"agg_max_org_design_with_SH_NC_{CASE_NAME}.pkl")
path_agg_min_with_SH_NC = os.path.join(DIR_DATA_FINAL, f"agg_min_org_design_with_SH_NC_{CASE_NAME}.pkl")

path_final_df_BC_wOutlier = os.path.join(DIR_DATA_FINAL, f"final_df_BC_wOutlier_{CASE_NAME}.pkl")
path_final_df_NC_wOutlier = os.path.join(DIR_DATA_FINAL, f"final_df_NC_wOutlier_{CASE_NAME}.pkl")

path_final_df_BC = os.path.join(DIR_DATA_FINAL, f"final_df_BC_{CASE_NAME}.pkl")
path_final_df_NC = os.path.join(DIR_DATA_FINAL, f"final_df_NC_{CASE_NAME}.pkl")


#%% 1st Work Flow - Part A: EPC data and radiator nominal values
print("Starting 1st workflow. Part A: Obtaining EPC data and Radiator nominal values")

# ---------------------------------------------------------------------- #
# Retrieve EPC Serial Identifiers for each property
# ---------------------------------------------------------------------- #
from A1_get_EPC_Identifier import retrieve_EPC

retrieve_EPC(stamdata=master_ids,
              EMAIL=EMAIL, PASSWORD=PASSWORD,
              path_epc_id=path_epc_id)

# ---------------------------------------------------------------------- #
# Retrieve the full EPC XML report for each EPC Serial Identifier
# ---------------------------------------------------------------------- #
from A2_get_EPC_XML import retrieve_EPC_XML

retrieve_EPC_XML(path_epc_id=path_epc_id,
                  path_epc_xml=path_epc_xml,
                  EMAIL=EMAIL, PASSWORD=PASSWORD)

# ---------------------------------------------------------------------- #
# Add historical design U-values and temperature sets to the EPC dataset
# ---------------------------------------------------------------------- #
from A3_get_design_values import get_design_values

get_design_values(path_load_data=path_epc_xml,
                   path_save_data=path_epc_xml)

# ---------------------------------------------------------------------- #
# Calculate design heat losses using the EPC and historical design U-values
# ---------------------------------------------------------------------- #
from A4_get_heatloss import get_heatloss

get_heatloss(path_load_data=path_epc_xml,
              path_save_data=path_epc_xml)

# ---------------------------------------------------------------------- #
# Calculate nominal radiator parameters and EPC-based supply/return temperatures
# ---------------------------------------------------------------------- #
from A5_get_DHTemperatures import get_DHTemperatures

get_DHTemperatures(path_load_data=path_epc_xml,
                   path_save_data=path_epc_xml,
                   n = Radn,
                   fs = fs)

# ---------------------------------------------------------------------- #
# Extract the BBR (building registry) data into a flat table
# ---------------------------------------------------------------------- #
from A6_get_BBRfromXML import get_BBRfromXML

get_BBRfromXML(path_load_data=path_epc_xml,
                path_save_data=path_bbr_xml)

# ---------------------------------------------------------------------- #
# Extract the design condition / radiator parameter data into flat tables
# ---------------------------------------------------------------------- #
from A7_get_heatCalcfromXML import get_heatCalcfromXML_optimized

get_heatCalcfromXML_optimized(path_load_data=path_epc_xml,
                               path_save_data_current=path_current_design,
                               path_save_data_max=path_max_design,
                               path_save_data_min=path_min_design,
                               show_progress=True)

# ---------------------------------------------------------------------- #
# Aggregate the EPC dataset onto the user-defined component IDs
# ---------------------------------------------------------------------- #
from A8_get_aggXML import get_aggXMLs

get_aggXMLs(path_bbr_xml=path_bbr_xml,
            compID=compID,
            path_agg_bbr=path_agg_bbr,
            path_current_design=path_current_design,
            path_agg_current=path_agg_current,
            path_max_design=path_max_design,
            path_agg_max=path_agg_max,
            path_min_design=path_min_design,
            path_agg_min=path_agg_min)

print("1st workflow completed successfully.")


#%% 2nd Work Flow - Part B: Predicted peak space heating consumption
print("Starting 2nd workflow. Part B: Obtaining peak space heating consumption")

# ---------------------------------------------------------------------- #
# Process the raw hourly meter data (drop redundant columns, clean data)
# ---------------------------------------------------------------------- #
from B1_get_processedHM import proc_raw_meter

proc_raw_meter(path_meter=path_meter_year1,
                path_save_meter=path_proc_meter_year1,
                grouping_columns=meter_id_list,
                year=YEAR1)

if YEAR2 is not None:
    proc_raw_meter(path_meter=path_meter_year2,
                    path_save_meter=path_proc_meter_year2,
                    grouping_columns=meter_id_list,
                    year=YEAR2)

# ---------------------------------------------------------------------- #
# Disaggregate total heat consumption into space heating (SH) and domestic
# hot water (DHW)
# ---------------------------------------------------------------------- #
from B2_get_disaggregated import apply_disagg

apply_disagg(file=path_proc_meter_year1,
              file_save=path_SH_year1,
              on_cols=meter_id_list)

if YEAR2 is not None:
    apply_disagg(file=path_proc_meter_year2,
                  file_save=path_SH_year2,
                  on_cols=meter_id_list)

# ---------------------------------------------------------------------- #
# Get hourly weather data for each year
# ---------------------------------------------------------------------- #
from B3_get_weather_data import get_weather

get_weather(path_weather_save=path_weather_year1,
            year=YEAR1,
            station_temperature=STATION_TEMPERATURE,
            station_wind_direction=STATION_WIND,
            station_wind_speed=STATION_WIND,
            station_solar_radiation=STATION_SOLAR_RADIATION)

if YEAR2 is not None:
    get_weather(path_weather_save=path_weather_year2,
                year=YEAR2,
                station_temperature=STATION_TEMPERATURE,
                station_wind_direction=STATION_WIND,
                station_wind_speed=STATION_WIND,
                station_solar_radiation=STATION_SOLAR_RADIATION)

# ---------------------------------------------------------------------- #
# Cross-year quality check (only relevant if two years of meter data are
# available; skipped automatically if YEAR2 is None)
# ---------------------------------------------------------------------- #
from B4_get_addProcessHM import cross_year_check

cross_year_check(path_SH_year1=path_SH_year1,
                  path_weather_year1=path_weather_year1,
                  path_save_meter_year1=path_proc_SH_year1,
                  YEAR1=str(YEAR1),
                  path_SH_year2=path_SH_year2,
                  path_weather_year2=path_weather_year2,
                  path_save_meter_year2=path_proc_SH_year2,
                  YEAR2=str(YEAR2) if YEAR2 is not None else None)

# ---------------------------------------------------------------------- #
# Build the daily timeseries dataset for model training and testing
# ---------------------------------------------------------------------- #
from B5_get_timeseries import get_timeseries

get_timeseries(dataYR1=pd.read_parquet(path_proc_SH_year1, engine='pyarrow'),
                weatherY1=pd.read_parquet(path_weather_year1, engine='pyarrow'),
                path_save_data=path_timeseries,
                dataYR2=pd.read_parquet(path_proc_SH_year2, engine='pyarrow') if YEAR2 is not None else None,
                weatherY2=pd.read_parquet(path_weather_year2, engine='pyarrow') if YEAR2 is not None else None)

# ---------------------------------------------------------------------- #
# Train the data-driven heat loss model and obtain accuracy metrics
# ---------------------------------------------------------------------- #
from B6_get_actualUAs import run_modelling_scripts

run_modelling_scripts(path_load_data=path_timeseries,
                       path_save_data=path_model_params,
                       on_cols=meter_id_list,
                       seed = RANDOM_STATE)

# ---------------------------------------------------------------------- #
# Predict space heating heat loss for the specified outdoor design conditions
# ---------------------------------------------------------------------- #
from B7_get_actualHeat import get_modelUAs, heatloss_calculation

heatloss_calculation(
    path_data_save=path_pred_heatloss,
    UAs=get_modelUAs(path_load_model=path_model_params, on_cols=meter_id_list),
    conditions=pd.DataFrame(list(itertools.product(Tin, Tout, Ws)), columns=['Tin', 'Tout', 'Ws']),
    on_cols=meter_id_list,
)

# ---------------------------------------------------------------------- #
# Predict normalized annual space heating energy consumption
# ---------------------------------------------------------------------- #
from B8_get_energy import get_annual_energy

get_annual_energy(file_path_weather=path_weather_year1,
                  file_path_model=path_model_params,
                  HD=HD_year1,
                  file_path_save=path_annual_SH_year1,
                  on_cols=meter_id_list)

if YEAR2 is not None:
    get_annual_energy(file_path_weather=path_weather_year2,
                      file_path_model=path_model_params,
                      HD=HD_year2,
                      file_path_save=path_annual_SH_year2,
                      on_cols=meter_id_list)

# ---------------------------------------------------------------------- #
# Aggregate the hourly meter data, predicted heat loss and predicted annual
# energy onto the user-defined component IDs
# ---------------------------------------------------------------------- #
from B9_get_aggHM import get_aggregate_df

#################### Hourly meter data #####################################
get_aggregate_df(path_meter_year1=path_proc_SH_year1,
                  path_meter_save_year1=path_agg_proc_SH_year1,
                  compID=compID, data_type="meter",
                  path_meter_year2=path_proc_SH_year2,
                  path_meter_save_year2=path_agg_proc_SH_year2)

#################### Heat loss #####################################
get_aggregate_df(path_meter_year1=path_pred_heatloss,
                  path_meter_save_year1=path_agg_pred_heatloss,
                  compID=compID, data_type="heatloss")

#################### Annual SH energy #####################################
get_aggregate_df(path_meter_year1=path_annual_SH_year1,
                  path_meter_save_year1=path_agg_annual_SH_year1,
                  compID=compID, data_type="annual",
                  path_meter_year2=path_annual_SH_year2,
                  path_meter_save_year2=path_agg_annual_SH_year2)

print("2nd workflow completed successfully.")


#%% 3rd Work Flow - Part C: Minimum supply and return temperatures
print("Starting 3rd workflow. Part C: Obtaining minimum supply and return temperature")

# ---------------------------------------------------------------------- #
# Calculate the minimum radiator supply and return temperatures
# ---------------------------------------------------------------------- #
from C1_get_actualDHTemperatures import get_LTDH_temperatures

# Building-level constraint
get_LTDH_temperatures(path_load_actHL=path_agg_pred_heatloss,
                       path_load_min=path_agg_min,
                       path_load_max=path_agg_max,
                       path_save_min=path_agg_min_with_SH_BC,
                       path_save_max=path_agg_max_with_SH_BC,
                       path_load_BBR=None,
                       col_HL='TotalHeatloss',
                       method='BC',
                       Tinb=Tin_peak,
                       Wsb=None, 
                       Toutb=None)

# Network-level constraint
get_LTDH_temperatures(path_load_actHL=path_agg_pred_heatloss,
                       path_load_min=path_agg_min,
                       path_load_max=path_agg_max,
                       path_save_min=path_agg_min_with_SH_NC,
                       path_save_max=path_agg_max_with_SH_NC,
                       path_load_BBR=path_agg_bbr,
                       col_HL='TotalHeatloss',
                       method='NC',
                       Tinb=Tin_peak,
                       Wsb=Ws_peak, 
                       Toutb=Tout_peak)

# ---------------------------------------------------------------------- #
# Identify and tag outliers, then select the relevant columns
# ---------------------------------------------------------------------- #
from C2_get_df_visuals import df_visuals, tag_outliers, wrangle_df

# Building-level Constraint
df_visuals(path_load_max=path_agg_max_with_SH_BC,
           path_load_min=path_agg_min_with_SH_BC,
           path_load_current=path_agg_current,
           path_save_HL=path_final_df_BC_wOutlier,
           NC=False)

tag_outliers(path_main_df=path_final_df_BC_wOutlier,
              path_load_BBR=path_agg_bbr,
              path_annual_SH_YR1=path_agg_annual_SH_year1,
              path_save_data=path_final_df_BC_wOutlier,   
              path_annual_SH_YR2=path_agg_annual_SH_year2, 
              lower_limit = lower_limit,
              upper_limit = upper_limit)

# Final data wrangle (removing outliers and consolidating EPC rates)
wrangle_df(path_load_data=path_final_df_BC_wOutlier,
           path_load_buildingdata=path_agg_bbr,
           path_load_energy=path_agg_annual_SH_year1,
           path_save_data=path_final_df_BC,
           Tinb=Tin_peak,
           Toutb=Tout_peak,
           col_build=None,
           removeOutliers=True)

# Network-level Constraint
df_visuals(path_load_max=path_agg_max_with_SH_NC,
           path_load_min=path_agg_min_with_SH_NC,
           path_load_current=path_agg_current,
           path_save_HL=path_final_df_NC_wOutlier,
           NC=True)

tag_outliers(path_main_df=path_final_df_NC_wOutlier,
              path_load_BBR=path_agg_bbr,
              path_annual_SH_YR1=path_agg_annual_SH_year1,
              path_save_data=path_final_df_NC_wOutlier,
              path_annual_SH_YR2=path_agg_annual_SH_year2, 
              lower_limit = lower_limit,
              upper_limit = upper_limit)

# Final data wrangle (removing outliers and consolidating EPC rates)
wrangle_df(path_load_data=path_final_df_NC_wOutlier,
           path_load_buildingdata=path_agg_bbr,
           path_load_energy=path_agg_annual_SH_year1,
           path_save_data=path_final_df_NC,
           Tinb=Tin_peak,
           Toutb=Tout_peak,
           col_build=None,
           removeOutliers=True)

print("3rd workflow completed successfully.")


#%% Get the final dataset ready for analysis
# ---------------------------------------------------------------------- #
from C2_get_df_visuals import adjust_nominal_values, map_plot

# Select the dataframe for the chosen method
if get_BC:
    path_final_use = path_final_df_BC
    var_to_select = var_BC
else:
    path_final_use = path_final_df_NC
    var_to_select = var_NC

selected_var = var_to_select[ind]

df = adjust_nominal_values(path_data=path_final_use,
                            fsb=fs,
                            Wsb=Ws,
                            Tinb=Tin,
                            Toutb=Tout,
                            paramb=paramb,
                            var=selected_var)

# Apply the heat exchanger temperature offset to supply/return temperatures
if "Tsupply" in selected_var or "Treturn" in selected_var:
    df[selected_var] = df[selected_var] + temp_HEX

with open(path_agg_bbr, 'rb') as f:
    buildingdata = pickle.load(f)

# Plot a map of the selected metric
map_plot(df_temp=adjust_nominal_values(path_data=path_final_use,
                                        fsb=[fs[ifs]],
                                        Wsb=[Ws_peak],
                                        Tinb=[Tin_peak],
                                        Toutb=[Tout_peak],
                                        paramb=["Coef"],
                                        var=var_to_select[1]),
         df_nom_heatloss=adjust_nominal_values(path_data=path_final_use,
                                                fsb=[fs[ifs]],
                                                Wsb=[Ws_peak],
                                                Tinb=[Tin_peak],
                                                Toutb=[Tout_peak],
                                                paramb=["Coef"],
                                                var=var_to_select[4]),
         LTDH_lvl=70,
         MTDH_lvl=90,
         buildingdata=buildingdata,
         fig_save_name=f"{CASE_NAME}_map_plot",
         title_plot=(f"LTDH: <70\u00b0C, MTDH: 70-90\u00b0C. Tin = {Tin_peak}\u00b0C, "
                      f"Tout = {Tout_peak}\u00b0C, Ws = {Ws_peak} m/s, no sun. "
                      f"Radiator exponent = {Radn}."
                      f"Sizing factor = {fs[ifs]}."),
         path_ids=compID)

#%% Final datasets ready for analysis
# ---------------------------------------------------------------------- #
# path_final_df_BC and path_final_df_NC contain the final, per-component
# datasets (with outliers removed) for the "BC" and "NC" methods
# respectively. Use these for further analysis or custom plotting.
