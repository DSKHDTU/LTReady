# -*- coding: utf-8 -*-
"""
C2_get_df_visuals.py

Prepares the final dataset for analysis and visualization, by:
    - merging the minimum/maximum/current design heat loss and supply/return
      temperature tables into a single long-format dataframe (df_visuals());
    - flagging outlier components based on their predicted annual SH energy
      use relative to similar-age buildings (find_outlier_energy(),
      tag_outliers());
    - removing outliers and merging in building/energy data (wrangle_df());
    - selecting a specific radiator exponent / wind speed / indoor-outdoor
      design temperature combination for analysis (adjust_nominal_values());
    - plotting the result on a map (map_plot()).
"""

import pandas as pd
import pickle
import numpy as np

import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

from A8_get_aggXML import prefix_name_to_column

def find_outlier_energy(path_annual_SH: str, buildingdata: pd.DataFrame, lower_limit: float, upper_limit: float) -> pd.DataFrame:
    """
    Flag components whose predicted annual SH energy use (kWh/m2) is an
    outlier relative to other components of similar construction age.

    Outlier detection is performed by comparing kWh/m2 against the 5th and
    99th percentile values within two construction-year groups (before 1960
    and from 1960 onwards), since consumption patterns differ markedly
    between these periods.

    For components below the lower (lower_limit) bound:
        - Likely missing meter/address coverage, underestimating consumption.
    For components above the upper (upper_limit) bound:
        - Likely missing EPC data for some buildings, overestimating
          consumption per m2.

    Components are also flagged as outliers if fewer than 10% of their
    buildings are residential (UseCode < 200), or if none of their buildings
    have district heating.

    path_annual_SH: str
        File path (.parquet) of the aggregated predicted annual SH energy for
        a given year (output of B9_get_aggHM.py, data_type="annual").
    buildingdata: pd.DataFrame
        Aggregated BBR dataframe (output of A8_get_aggXML.py, aggBBR()),
        containing "component_id", "YearOfConstruction", "FloorArea",
        "hasDH", "NumberOfBuildings", "UseCodeList" and "EPC_Class_Property".
    lower_limit: float
        Lower percentile value used in outlier detection.
    upper_limit: float
        Upper percentile value used in outlier detection.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "component_id" and "outlier" (bool), listing
        all components flagged as outliers.
    """
    # Get the meter data
    df = pd.read_parquet(path_annual_SH, engine='pyarrow')
    
    # Merge with dataframe containing construction year and floor area
    df = pd.merge(df, buildingdata[['component_id','YearOfConstruction','FloorArea','hasDH','NumberOfBuildings','UseCodeList','EPC_Class_Property']], how='left', on = ['component_id'])
        
    # Remove component if majority of buildings are not housing 
    def pctUseCode(lst, useCode = 200):
        """
        Calculate the percentage of usecodes below 200 in a list.
        (usecode < 200 is residential/housing) (https://teknik.bbr.dk/kodelister/0/1/0/BygAnvendelse)
        
        Returns:
            float between 0 and 100, or np.nan if input is empty or not a list
        """
    
        if not isinstance(lst, list) or len(lst) == 0:
            return np.nan
        
        total = len(lst)
        count = sum(1 for x in lst if x < useCode)
        
        return (count / total) * 100
    
    df['pctHouse'] = df['UseCodeList'].apply(pctUseCode)
    
    dfreg = df[df['pctHouse']>10]
    
    # Remove buildings without DH
    dfreg = dfreg[dfreg['hasDH'].apply(any)]
    
    # Prepare the variable for outlier detection
    dfreg['kWh/m2'] = dfreg['Energy_MWh']*1000 / dfreg['FloorArea']
    
    # Get the specific construction period and their limits
    limits_b1960 = dfreg.query("YearOfConstruction<1960")['kWh/m2'].quantile([lower_limit, upper_limit])
    limits_a1960 = dfreg.query("YearOfConstruction>=1960")['kWh/m2'].quantile([lower_limit, upper_limit])

    # Outliers for buildings before 1960
    outliers_b1960 = dfreg.query("YearOfConstruction < 1960").loc[
        (dfreg['kWh/m2'] < limits_b1960[lower_limit]) | (dfreg['kWh/m2'] > limits_b1960[upper_limit]),
        'component_id'
    ]

    # Outliers for buildings from 1960 and later
    outliers_a1960 = dfreg.query("YearOfConstruction >= 1960").loc[
        (dfreg['kWh/m2'] < limits_a1960[lower_limit]) | (dfreg['kWh/m2'] > limits_a1960[upper_limit]),
        'component_id'
    ]
    
    # Combine
    id_outlier = pd.concat([outliers_b1960, outliers_a1960], axis=0).reset_index(drop=True)
        
    # Get the list of BFEnummer which must be removed
    id_outlier = id_outlier.reset_index(drop=True).to_frame()
    id_outlier['outlier'] = True
    
    # Add additional outlier tags based on usecode and hasDH
    df_out = df[(df['pctHouse'] <= 10)| (~df['hasDH'].apply(any))]
    df_out['outlier'] = True
    df_out = df_out[['component_id','outlier']].drop_duplicates()
    
    # Combine
    id_outlier = id_outlier.merge(df_out, on = ['component_id','outlier'], how = 'outer')
              
    return id_outlier    

def df_visuals(path_load_max: str, path_load_min: str, path_load_current: str, path_save_HL: str, deltaT: bool = False) -> None:
    """
    Combine the maximum/minimum/current design heat loss and supply/return
    temperature tables into a single long-format ("variable"/"value")
    dataframe and save it.

    path_load_max: str
        File path (.parquet) of the result based on the maximum
        original-design table (output of C1_get_actualDHTemperatures.py,
        `path_save_max`).
    path_load_min: str
        File path (.parquet) of the result based on the minimum
        original-design table (output of C1_get_actualDHTemperatures.py,
        `path_save_min`).
    path_load_current: str
        File path (.pkl) of the aggregated current-design heat loss table
        (output of A8_get_aggXML.py, aggCurrent()).
    path_save_HL: str
        File path (.pkl) where the resulting long-format dataframe is saved.
    deltaT: bool, optional
        Set to True when processing the "deltaT" method output (drops the
        duplicated "Actual Design TotalHeatloss"/"Actual Design massFlow"
        columns from the maximum-design table). Default False.

    Returns
    -------
    None
        The resulting dataframe is written to `path_save_HL` as a pickle file.
    """
    # Upload consumption and design data
    HL_max  = pd.read_parquet(path_load_max, engine='pyarrow')
    HL_max.rename(columns = {'Actual Tsupply': 'Max Actual Tsupply', 'Actual Treturn': 'Max Actual Treturn'}, inplace = True)
    
    HL_min = pd.read_parquet(path_load_min, engine='pyarrow')
    HL_min.rename(columns = {'Actual Tsupply': 'Min Actual Tsupply', 'Actual Treturn': 'Min Actual Treturn'}, inplace = True)
        
    # Keywords to match (removes any column containing these words)
    keywords_to_remove = ['HeatlossInclGain', 'InclMechanicalVentilationLoss', 'InternalHeatGain','BFEnummer','EPCSerial','LabelStatus','BuildingNumber',
                          'windows', 'buildingparts', 'linelosses','ventilation','T design supply','T design return','Tin design']
    
    # Filter out columns that contain any of the keywords
    HL_max = HL_max[[col for col in HL_max.columns if not any(keyword in col for keyword in keywords_to_remove)]]
    HL_min = HL_min[[col for col in HL_min.columns if not any(keyword in col for keyword in keywords_to_remove)]]
    
    if deltaT:
        # Remove Actual Design heat loss and massflow from one of the HLs as its the same for both df
        HL_max.drop(columns=['Actual Design TotalHeatloss','Actual Design massFlow'], inplace = True)
    
    # Convert Index to Series, Convert to DataFrames and perform an inner merge
    col_share = pd.merge(pd.Series(HL_max.columns, name="col"), 
                         pd.Series(HL_min.columns, name="col"), 
                         on="col", how="inner")['col'].tolist()
        
    # Melt the data frame - max 
    col_df = pd.Series(HL_max.columns, name = 'col').to_frame()
    HL_max = HL_max.melt(value_vars = col_df.query('~col.isin(@col_share)')['col'].tolist(),
                         id_vars = col_share,
                         value_name = 'value', var_name = 'variable')
    
    # Melt the data frame - min
    col_df = pd.Series(HL_min.columns, name = 'col').to_frame()
    HL_min = HL_min.melt(value_vars = col_df.query('~col.isin(@col_share)')['col'].tolist(),
                         id_vars = col_share,
                         value_name = 'value', var_name = 'variable')
    
    # Merge the upper and lower dataframes
    col_share = pd.merge(pd.Series(HL_max.columns, name="col"), 
                         pd.Series(HL_min.columns, name="col"), 
                         on="col", how="inner")['col'].tolist()
        
    # Merge the dataframe
    final_df = pd.merge(HL_max, HL_min, how = 'outer',on = col_share)
    
    if path_load_current is not None:
        with open(path_load_current, 'rb') as f:
            HL_cur = pickle.load(f) 
            
        del f
        HL_cur.drop(columns = ['BFEnummer', 'EPCSerial', 'LabelStatus','BuildingNumber','NumberOfBuildings','InternalHeatGain', 'HeatlossInclGain'], inplace = True)
        HL_cur = prefix_name_to_column(HL_cur, 'Current ', ['component_id'])
        
        # Add the HL_cur
        final_df = pd.merge(final_df, HL_cur, how = 'outer',on = 'component_id')

    # Save the relevant data
    with open(path_save_HL, 'wb') as f:
        pickle.dump(final_df, f)


def tag_outliers(path_main_df: str, path_load_BBR: str, path_annual_SH_YR1: str,
                  path_save_data: str, path_annual_SH_YR2: str = None, 
                  lower_limit: float = 0.05, upper_limit: float = 0.99) -> None:
    """
    Tag each component as an outlier or not, based on its predicted annual SH
    energy use (see find_outlier_energy()), and save the result.

    If two years of annual SH energy data are provided, a component is
    flagged as an outlier if it is flagged in either year.

    path_main_df: str
        File path (.pkl) of the long-format dataframe (output of df_visuals()).
    path_load_BBR: str
        File path (.pkl) of the aggregated BBR table (output of
        A8_get_aggXML.py, aggBBR()).
    path_annual_SH_YR1: str
        File path (.parquet) of the aggregated predicted annual SH energy for
        year 1 (output of B9_get_aggHM.py, data_type="annual").
    path_save_data: str
        File path (.pkl) where the result is saved.
    path_annual_SH_YR2: str, optional
        File path (.parquet) of the aggregated predicted annual SH energy for
        year 2. If None (default), outlier tagging is based on year 1 only.
    lower_limit: float
        Lower percentile value used in outlier detection.
    upper_limit: float
        Upper percentile value used in outlier detection.

    Returns
    -------
    None
        `path_main_df` with an added boolean "outlier" column is written to
        `path_save_data` as a pickle file.
    """
    # Tag the outliers based on find_outlier_energy()
    ## For some components we only have annual SH data for one year, so we
    ## merge across both years (if available) to fill in tags for either.
    with open(path_main_df, 'rb') as f:
        final_df = pickle.load(f)
        
    with open(path_load_BBR, 'rb') as f:
        buildingdata = pickle.load(f) 

    ## Based on year 1 data
    id_outlier = find_outlier_energy(path_annual_SH_YR1, buildingdata,lower_limit,upper_limit)

    if path_annual_SH_YR2 is not None:
        ## Based on year 2 data
        idYR2 = find_outlier_energy(path_annual_SH_YR2, buildingdata,lower_limit,upper_limit)

        # Merge
        id_outlier = id_outlier.merge(idYR2, how='outer', on=['component_id','outlier'])
        
    # Get the final dataframe with outliers removed
    final_df = pd.merge(final_df, id_outlier, on = ['component_id'], how = 'left')
    final_df["outlier"] = final_df["outlier"].fillna(False)

    # Save the relevant data
    with open(path_save_data, 'wb') as f:
        pickle.dump(final_df, f)
        
def toggles_to_adj_capacity(sel: pd.DataFrame, Tinb: float, Toutb: float) -> pd.DataFrame:
    """
    Determine, for each component, whether the minimum or maximum
    original-design heat loss should be used to represent the current
    (EPC-based) heat loss, by comparing the temperature-normalized current
    heat loss to the temperature-normalized minimum design heat loss.

    sel: pd.DataFrame
        Long-format dataframe (output of df_visuals()/tag_outliers()), with
        "component_id", "Current TotalHeatloss", "variable" and "value" columns.
    Tinb: float
        Indoor design temperature [degC] used to normalize the current heat loss.
    Toutb: float
        Outdoor design temperature [degC] used to normalize the current heat loss.

    Returns
    -------
    pd.DataFrame
        One row per component_id, with boolean "useMax"/"useMin" columns
        (mutually exclusive) indicating which original-design heat loss
        range is closer to the current EPC-based heat loss.
    """
    # Provide design values for Tin and Tout used in calculating the current totalheatloss (total heatloss based on EPC U values)
    
    df = sel.query('variable.isin(["Min TotalHeatloss","Min Tout design"])')[['component_id','Current TotalHeatloss','variable','value']].drop_duplicates()

    # Pivot temp
    df = df.pivot(index = ['component_id','Current TotalHeatloss'], columns = 'variable', values = 'value').reset_index()

    # Normalize the consumptions to same temperatures
    df['Norm Current TotalHeatloss'] = (df['Current TotalHeatloss']/(Tinb-(Toutb))) 
    df['Norm Min TotalHeatloss'] = (df['Min TotalHeatloss']/(Tinb-df['Min Tout design']))

    # Toggle for using either min or max value
    df['useMax'] = np.where(df['Norm Current TotalHeatloss'] / df['Norm Min TotalHeatloss'] > 1, 1, 0)
    df['useMin'] = np.where(df['Norm Current TotalHeatloss'] / df['Norm Min TotalHeatloss'] <= 1, 1, 0)
    
    # Set the toggles as a dataframe
    df = df[['component_id','useMax','useMin']].drop_duplicates()  
    
    return df

def rename_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the public-facing naming convention to the final dataset.

    Renames the literal "TotalHeatloss"/"Current TotalHeatloss" columns,
    maps the long-format "variable" column's string values to their new
    names, and replaces any remaining spaces in column names with
    underscores.

    df: pd.DataFrame
        Final dataframe (output of the outlier-removal step in wrangle_df()),
        containing a "variable" column plus the literal columns produced by
        df_visuals()/tag_outliers().

    Returns
    -------
    pd.DataFrame
        `df` with renamed columns and renamed "variable" values.
    """
    # Rename literal columns
    column_renames = {
        'TotalHeatloss': 'Predicted_Current_Totalheatloss',
        'Current TotalHeatloss': 'EPC_Current_Totalheatloss',
    }
    df = df.rename(columns=column_renames)

    # Rename the values within the "variable" column
    variable_renames = {
        'Min TotalHeatloss': 'Lower_Historical_Nominal_Totalheatloss',
        'Max TotalHeatloss': 'Upper_Historical_Nominal_Totalheatloss',
        'Min K': 'Lower_Radiator_K',
        'Max K': 'Upper_Radiator_K',
        'Min Tout design': 'Lower_Historical_Nominal_Tout',
        'Max Tout design': 'Upper_Historical_Nominal_Tout',
        'Min Design massFlow': 'Lower_Historical_Nominal_Massflow',
        'Max Design massFlow': 'Upper_Historical_Nominal_Massflow',
        'Min Actual Tsupply': 'Lower_Minimum_Tsupply',
        'Max Actual Tsupply': 'Upper_Minimum_Tsupply',
        'Min Actual Treturn': 'Lower_Minimum_Treturn',
        'Max Actual Treturn': 'Upper_Minimum_Treturn',
        'Actual Design TotalHeatloss': 'Current_Nominal_Totalheatloss',
        'Actual Design massFlow': 'Current_Nominal_Massflow',
        'Min Tsupply': 'Lower_EPC_calculated_Tsupply',
        'Max Tsupply': 'Upper_EPC_calculated_Tsupply',
        'Min Treturn': 'Lower_EPC_calculated_Treturn',
        'Max Treturn': 'Upper_EPC_calculated_Treturn',
    }
    if 'variable' in df.columns:
        df['variable'] = df['variable'].replace(variable_renames)

    # Replace any remaining spaces in column names with underscores
    df.columns = [col.replace(' ', '_') for col in df.columns]

    return df

def wrangle_df(path_load_data: str, path_load_buildingdata: str, path_load_energy: str,
                path_save_data: str, Tinb: float, Toutb: float,
                col_build: list = None, removeOutliers: bool = True) -> None:
    """
    Final data wrangling step: merge in building/energy data, group EPC
    classes into broader performance bands, optionally remove outliers, and
    save the result.

    path_load_data: str
        File path (.pkl) of the long-format dataframe (output of
        tag_outliers()).
    path_load_buildingdata: str
        File path (.pkl) of the aggregated BBR table (output of
        A8_get_aggXML.py, aggBBR()).
    path_load_energy: str
        File path (.parquet) of the aggregated predicted annual SH energy
        (output of B9_get_aggHM.py, data_type="annual") used to compute kWh/m2.
    path_save_data: str
        File path (.pkl) where the final dataframe is saved.
    Tinb: float
        Indoor design temperature [degC] used to normalize the current heat loss.
    Toutb: float
        Outdoor design temperature [degC] used to normalize the current heat loss.
    col_build: list of str, optional
        Additional columns from `path_load_buildingdata` to merge into the
        result and use to compute "W/m2" (heat loss per floor area). If None
        or empty (default), no additional columns are merged and "W/m2" is
        not computed.
    removeOutliers: bool, optional
        If True (default), remove components flagged as outliers (see
        tag_outliers()) from both the result and the energy data before merging.

    Returns
    -------
    None
        The final dataframe is written to `path_save_data` as a pickle file.
        `path_load_buildingdata` is also overwritten with an added
        "Performance" column (EPC class grouped into "A-B"/"C-D"/"E-G"/"Unknown").
    """
    if col_build is None:
        col_build = []

    with open(path_load_data, 'rb') as f:
        df = pickle.load(f)
        
    with open(path_load_buildingdata, 'rb') as f:
        buildingdata = pickle.load(f)
    
    energy = pd.read_parquet(path_load_energy, engine="pyarrow")
    
    # Wrangle buildingdata
    mapping = {
        'A': 'A-B', 'B': 'A-B',
        'C': 'C-D', 'D': 'C-D',
        'E': 'E-G', 'F': 'E-G', 'G': 'E-G'
    }

    buildingdata['Performance'] = buildingdata['EPC_Class_Property'].map(mapping).fillna('Unknown')
    
    # Wrangle energydata
    energy = pd.merge(buildingdata[['component_id','FloorArea','Performance','EPC_Class_Property','UseCode','YearOfConstruction']], energy, on = 'component_id', how = 'outer')
    energy['kWh/m2'] = round(energy['Energy_MWh']*1000 / energy['FloorArea'],2)
    
    if not col_build:
        sel = df
    else:
        sel = df.merge(buildingdata[['component_id','FloorArea'] + col_build], on = 'component_id', how = 'left')
        
        # Set up W/m2
        sel['W/m2'] = sel['TotalHeatloss'] / sel['FloorArea']
        
    # Remove outliers (where construction year and annual energy doesnt match)
    if removeOutliers:
        id_outliers = sel.query('outlier==True')['component_id'].drop_duplicates()
        
        # Remove duplicates from sel
        sel.query('outlier==False', inplace = True)
        
        # Remove duplicates from energy
        energy = energy[~energy['component_id'].isin(id_outliers)]
        energy.dropna(inplace=True)
        
        # Add energy to sel
        sel = sel.merge(energy, how = 'left')
        
    # Add the capacity adjustment toggles (useMin/useMax) to the main dataframe
    sel = sel.merge(toggles_to_adj_capacity(sel, Tinb, Toutb), on = 'component_id', how = 'left')

    # Apply the public-facing naming convention
    sel = rename_final_columns(sel)
        
    # Save the relevant data
    with open(path_save_data, 'wb') as f:
        pickle.dump(sel, f)
        
    with open(path_load_buildingdata, 'wb') as f:
        pickle.dump(buildingdata, f)

def adjust_nominal_values(path_data: str, fsb: list, Wsb: list, Tinb: list, Toutb: list, paramb: list, var: str) -> pd.DataFrame:
    """
    Select a single radiator-sizing / wind-speed / indoor-and-outdoor
    design-temperature / parameter-statistic combination from the final
    dataset, and adjust the chosen variable using the minimum/maximum
    original-design capacity toggles (see toggles_to_adj_capacity()).

    path_data: str
        File path (.pkl) of the final dataframe (output of wrangle_df()).
    fsb: list of float
        Radiator sizing value(s) to select (e.g. [1.0]).
    Wsb: list of float
        Wind speed value(s) to select [m/s].
    Tinb: list of float
        Indoor design temperature value(s) to select [degC].
    Toutb: list of float
        Outdoor design temperature value(s) to select [degC].
    paramb: list of str
        Parameter statistic value(s) to select (e.g. ['Coef'], or
        ['Coef','Conf_low','Conf_up']).
    var: str
        Name of the variable to select and adjust (e.g. 'Tsupply',
        'TotalHeatloss'). The function looks for 'Lower' + var and
        'Upper' + var in the "variable" column.

    Returns
    -------
    pd.DataFrame
        One row per component_id (and selected combination), with a column
        named `var` containing the capacity-adjusted value (a weighted
        combination of 'Lower' + var and 'Upper' + var based on useMin/useMax).
    """
    # Load data
    with open(path_data, 'rb') as f:
        df = pickle.load(f)
    
    # Select relevant parameters
    df_adj = df.query('fs.isin(@fsb) & Ws.isin(@Wsb) & Tin.isin(@Tinb) & Tout.isin(@Toutb) & param_stats.isin(@paramb)')
    
    if var in ['Current_Nominal_Totalheatloss','Current_Nominal_Massflow','Predicted_Current_Totalheatloss','EPC_Current_Totalheatloss']:
        df_adj = df_adj.query('variable==@var')
    else:
        # Select the variables we want for the analysis    
        var_to_get = ['Lower_' + var, 'Upper_' + var]
        df_adj = df_adj.query('variable.isin(@var_to_get)')
    
        # Pivot
        colind = df_adj.drop(columns=['variable','value']).columns
        df_adj = df_adj.pivot(index=colind, columns='variable', values='value').reset_index()
        
        # Adjust variable with the useMin/useMax toggles
        df_adj[var] = df_adj[var_to_get[0]]*df_adj['useMin'] + df_adj[var_to_get[1]]*df_adj['useMax']
        
        # Remove toggles
        df_adj.drop(columns=['useMax','useMin'], inplace = True)
    
    return df_adj

def map_plot(df_temp: pd.DataFrame, df_nom_heatloss: pd.DataFrame, LTDH_lvl: float, MTDH_lvl: float,
             buildingdata: pd.DataFrame, fig_save_name: str, title_plot: str = "", path_ids: str = None) -> None:
    """
    Plot each component on an interactive map, colored by whether its
    estimated minimum supply temperature is compatible with low-temperature
    district heating (LTDH), medium-temperature district heating (MTDH), or
    neither, and save the figure as an HTML file.

    df_temp: pd.DataFrame
        Output of adjust_nominal_values() for a supply-temperature variable
        (e.g. var='Minimum_Tsupply'). Must contain
        "component_id" and a "Minimum_Tsupply" column (the variable column produced
        by adjust_nominal_values()).
    df_nom_heatloss: pd.DataFrame
        Output of adjust_nominal_values() for the nominal design heat loss
        (e.g. var='Predicted_Current_Totalheatloss'). Must contain "component_id" and
        "Historical_Nominal_Massflow".
    LTDH_lvl: float
        Maximum supply temperature [degC] still considered compatible with
        low-temperature district heating.
    MTDH_lvl: float
        Maximum supply temperature [degC] still considered compatible with
        medium-temperature district heating.
    buildingdata: pd.DataFrame
        Aggregated BBR table (or a subset of it) containing
        "EPC_Class_Property", "YearOfConstruction" and "FloorArea", indexed
        consistently with `df_temp`/`df_nom_heatloss` (merged on shared columns).
    fig_save_name: str
        Output file name (without extension); the figure is saved as
        "<fig_save_name>.html".
    title_plot: str, optional
        Title displayed on the map. Default "".
    path_ids: pd.DataFrame, optional
        File path (.pkl) of a dataframe with additional per-component
        identifiers/metadata (e.g. an internal reference ID, "bfe") to merge
        in for display in the hover tooltip. Default None (no extra IDs merged).

    Returns
    -------
    None
        Opens the figure in a browser tab and saves it as
        "<fig_save_name>.html" in the current working directory.
    """
    # Selecting the variables  
    df = df_temp.merge(buildingdata[['component_id','EPC_Class_Property', 'YearOfConstruction', 'FloorArea', 'Type of system','lat','lon']]).merge(df_nom_heatloss)

    # Turn unhashable list "type of system" to hashable to enable pivot
    df['Type_of_system'] = df['Type of system'].apply(lambda x: str(x))
        
    # Group based on performance
    df['Category'] = np.select(
        [
            df['Minimum_Tsupply'] <= LTDH_lvl,
            (df['Minimum_Tsupply'] < MTDH_lvl) & (df['Minimum_Tsupply'] > LTDH_lvl),
        ],
        ['1. LTDH Ready','2. MTDH Ready'],
        default='3. Not Ready'
    )

    # Radiator capacity
    df['Ratio_HL'] = round(df['Predicted_Current_Totalheatloss'] / df['Historical_Nominal_Totalheatloss'],2)

    # rounding
    df['Predicted_Peak_W/m2'] = round(df['Predicted_Current_Totalheatloss'] / df['FloorArea'],2)
    
    # rounding
    df['Minimum_Tsupply'] = round(df['Minimum_Tsupply'],2)

    if path_ids is not None:
        df = df.merge(path_ids[['component_id','bfe']], how = 'left')
        df['bfe'] = df['bfe'].apply(lambda x: str(x))

    fig = px.scatter_map(df, 
                         color = 'Category', lat = 'lat', lon = 'lon', 
                         zoom = 13,
                         hover_name='component_id',
                         hover_data = ['bfe','Predicted_Peak_W/m2','kWh/m2','Minimum_Tsupply','EPC_Class_Property','YearOfConstruction','FloorArea','Ratio_HL','Type_of_system'],
                         color_discrete_map={"1. LTDH Ready": 'rgb(27,120,55)', "2. MTDH Ready": 'rgb(252,141,89)', "3. Not Ready": 'rgb(215,48,39)', },
                         title=title_plot) 

    fig.show()

    fig.write_html(fig_save_name + ".html")