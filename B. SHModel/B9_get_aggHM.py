# -*- coding: utf-8 -*-
"""
B9_get_aggHM.py

Aggregates the per-meter hourly meter data, predicted heat loss, and predicted
annual SH energy onto the user-defined "component_id" grouping (where a
component may consist of one or more meters/buildings sharing a single
district heating substation).

For hourly meter data:
    - Consecutive missing values are removed after aggregation (these occur
      if not all meters within a component have data for a given hour).
    - If two years of data are provided, components where the number of
      meters differs between years are dropped from year 1 (the year with
      more meters is assumed more complete/recent).

Optimized for speed using vectorized sums, weighted averages, and
vectorized missing-streak detection.
"""

import pandas as pd
import numpy as np
import pickle
import ast

# Aggregation and component id
def getCmpID4Agg(df: pd.DataFrame, compID: pd.DataFrame) -> pd.DataFrame:
    """
    Add a "component_id" column to `df` by mapping its "serial" values to the
    user-defined component IDs in `compID`.

    df: pd.DataFrame
        Dataframe containing a "serial" column.
    compID: pd.DataFrame
        User-specified component ID mapping table. Must contain a
        "component_id" column and a "serial" column listing the meter
        serial(s) belonging to each component (as scalars, lists, or
        stringified lists).

    Returns
    -------
    pd.DataFrame
        `df` with an added "component_id" column.
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

    """Return one dict: serial_map mapping normalized id -> component_id"""
    serial_map = {}

    for _, row in compID.iterrows():
        comp = row['component_id']
        # serial
        if 'serial' in row:
            for b in _ensure_list(row['serial']):
                key = str(b).strip()
                if key:
                    serial_map[key] = comp
    
    # Component ID for serial
    def choose_component(row, serial_map = serial_map):
        
        return serial_map.get(row['serial'])
    
    # apply to your dataframe
    df['component_id'] = df.apply(choose_component, axis=1)
    
    return df 

# Vectorized helper functions
def custom_sum(series: pd.Series) -> float:
    """
    Sum a series, returning NaN if any value in the series is NaN (rather
    than skipping NaNs as pandas does by default).

    series: pd.Series
        Input series.

    Returns
    -------
    float
        The sum of `series`, or NaN if any element is missing.
    """
    return series.sum(min_count=len(series))


def findbad_IDs(df: pd.DataFrame, col: str = "SH") -> pd.Index:
    """
    Identify component_ids where the longest consecutive streak of missing
    values in `col` exceeds 50% of that component's total rows.

    df: pd.DataFrame
        Dataframe containing "component_id", "timestamp" and `col`.
    col: str, optional
        Column to check for missing-value streaks (default "SH").

    Returns
    -------
    pd.Index
        component_id values with an excessive missing-value streak. These
        are dropped from the aggregated meter data in wrang_proc_meter().
    """
    df = df.sort_values(["component_id", "timestamp"])
    missing = df[col].isna().astype(int)

    # consecutive run IDs
    grp = (missing != missing.shift()).cumsum()

    # streak lengths
    streaks = df.groupby(["component_id", grp])[col].size().reset_index(name="streak")
    max_streaks = streaks.groupby("component_id")["streak"].max()

    # total counts
    counts = df.groupby("component_id").size()

    bad = max_streaks[max_streaks > 0.5 * counts].index
    return bad

# Aggregators
def agg_meter_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly meter data per (component_id, timestamp), summing
    energy/volume/heat-power/SH/DHW values and computing floor-area-free,
    heat-power-weighted average inlet/return flow temperatures.

    data: pd.DataFrame
        Hourly meter data with "component_id", "timestamp", "energy",
        "volume", "volume_flow", "heat_power", "SH", "DHW",
        "inlet_flow_temperature", "return_flow_temperature" and "serial".

    Returns
    -------
    pd.DataFrame
        One row per (component_id, timestamp), with summed quantities,
        weighted-average temperatures, and "n_meter" (number of meters
        contributing to that row).
    """
    df = data.copy()

    # Precompute weighted sums
    df["inlet_x_power"] = df["inlet_flow_temperature"] * df["heat_power"]
    df["return_x_power"] = df["return_flow_temperature"] * df["heat_power"]

    aggregated = (
        df.groupby(["component_id", "timestamp"], as_index=False)
        .agg(
            energy=("energy", custom_sum),
            volume=("volume", custom_sum),
            volume_flow=("volume_flow", custom_sum),
            heat_power=("heat_power", custom_sum),
            SH=("SH", custom_sum),
            DHW=("DHW", custom_sum),
            inlet_x_power=("inlet_x_power", "sum"),
            return_x_power=("return_x_power", "sum"),
            heat_power_sum=("heat_power", "sum"),
            n_meter=("serial", "count"),
        )
    )

    # Compute weighted averages
    aggregated["inlet_flow_temperature"] = aggregated["inlet_x_power"] / aggregated["heat_power_sum"]
    aggregated["return_flow_temperature"] = aggregated["return_x_power"] / aggregated["heat_power_sum"]

    aggregated.drop(columns=["inlet_x_power", "return_x_power", "heat_power_sum"], inplace=True)

    return aggregated

def agg_annual_SH(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate predicted annual SH energy per component_id.

    data: pd.DataFrame
        Annual energy data with "component_id", "Energy_MWh" and "serial".

    Returns
    -------
    pd.DataFrame
        One row per component_id, with summed "Energy_MWh" and "n_meter"
        (number of meters contributing).
    """
    df = data.copy()

    aggregated = (
        df.groupby(["component_id"], as_index=False)
        .agg(
            Energy_MWh=("Energy_MWh", custom_sum),
            n_meter=("serial", "count"),
        )
    )
    return aggregated


def agg_heatloss_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate predicted heat loss per (component_id, Tin, Tout, Ws, param_stats).

    data: pd.DataFrame
        Predicted heat loss data with "component_id", "Tin", "Tout", "Ws",
        "param_stats", "TotalHeatloss" and "serial".

    Returns
    -------
    pd.DataFrame
        One row per (component_id, Tin, Tout, Ws, param_stats), with summed
        "TotalHeatloss" and "n_meter" (number of meters contributing).
    """
    df = data.copy()

    aggregated = (
        df.groupby(["component_id", "Tin", "Tout", "Ws","param_stats"], as_index=False)
        .agg(TotalHeatloss=("TotalHeatloss", custom_sum),
             n_meter=("serial", "count"),
        )
    )
    return aggregated

# Wrangling
def wrang_proc_meter(data: pd.DataFrame, compID: pd.DataFrame, data_type: str = "meter") -> pd.DataFrame:
    """
    Aggregate `data` onto the user-defined "component_id" grouping.

    Components consisting of a single meter (compID["n_serial"] == 1) are
    passed through unchanged (aside from adding "component_id" and
    "n_meter"); components with multiple meters are aggregated using
    agg_meter_data(), agg_heatloss_data() or agg_annual_SH() depending on
    `data_type`.

    data: pd.DataFrame
        Input dataframe. Must contain a "serial" column, plus the columns
        required by the relevant aggregator (see agg_meter_data(),
        agg_heatloss_data(), agg_annual_SH()).
    compID: pd.DataFrame
        User-specified component ID mapping table. Must contain
        "component_id", "serial" and "n_serial" (number of meters per
        component) columns.
    data_type: str, optional
        One of "meter", "heatloss" or "annual" (default "meter").

    Returns
    -------
    pd.DataFrame
        `data` aggregated onto "component_id", with "serial" dropped and
        "n_meter" added. For data_type="meter", components with an
        excessive missing-value streak (see findbad_IDs()) are removed.
    """
    # Remove columns we are not aggregating or using for aggregation
    if data_type == "heatloss":
        data = data[["serial", "Tin", "Tout", "Ws","param_stats","TotalHeatloss"]]
    elif data_type == "annual":
        data = data[["serial", "Energy_MWh"]]
    elif data_type == "meter":
        data = data
    else:
        raise ValueError("Wrong data_type provided")
     
    # Identify the serials we have to aggregate and don't aggregate
    singleSerials = compID.loc[compID["n_serial"] == 1, ["component_id", "serial"]].explode("serial", ignore_index=True)
    
    # Dataframe incl. only serials to aggregate
    aggregated = data[~data['serial'].isin(singleSerials['serial'])]
    
    # Add the component ID to the dataframe
    aggregated = getCmpID4Agg(aggregated, compID)

    # Aggregation
    if data_type == "meter":
        aggregated = agg_meter_data(aggregated)
    elif data_type == "heatloss":
        aggregated = agg_heatloss_data(aggregated)
    elif data_type == "annual":
        aggregated = agg_annual_SH(aggregated)
    else:
        raise ValueError("Wrong data_type provided")

    # Add the component ID to the single-meter dataframe
    non_agg = data[data['serial'].isin(singleSerials['serial'])]
    non_agg = non_agg.merge(singleSerials, on = ['serial'], how = 'left')

    # Format the shape of the dataframe
    if data_type == "meter":
        non_agg["n_meter"] = non_agg.groupby(["component_id", "timestamp"])["serial"].transform("count")
    elif data_type == "heatloss":
        non_agg["n_meter"] = non_agg.groupby(["component_id", "Tin", "Tout", "Ws","param_stats"])["serial"].transform("count")
    elif data_type == "annual":
        non_agg["n_meter"] = non_agg.groupby(["component_id"])["serial"].transform("count")

    # Merge
    joint = pd.concat([non_agg, aggregated], ignore_index=True)

    # Drop redundant cols
    joint = joint.drop(columns=[c for c in ["serial"] if c in joint.columns])

    # Remove BFEs with too many missing streaks
    if data_type == "meter":
        bad_IDs = findbad_IDs(joint, col="SH")
        joint = joint[~joint["component_id"].isin(bad_IDs)]

    return joint

def keep_BFEs(data: pd.DataFrame, df: pd.DataFrame, bool_col: str) -> pd.DataFrame:
    """
    Keep only the rows of `data` whose "component_id" appears in `df` where
    `bool_col` is True.

    data: pd.DataFrame
        Dataframe to filter. Must contain "component_id".
    df: pd.DataFrame
        Dataframe with "component_id" and `bool_col` columns.
    bool_col: str
        Name of the boolean column in `df` used to select components to keep.

    Returns
    -------
    pd.DataFrame
        Rows of `data` whose "component_id" is flagged True in `df`.
    """
    keep = df.query(f"{bool_col}")["component_id"]
    return data[data["component_id"].isin(keep)]


def filter_unequal_meters(dataY1: pd.DataFrame, dataY2: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only components in `dataY1` whose number of meters ("n_meter")
    matches between `dataY1` and `dataY2`.

    This handles the case where a component had a different number of
    (active) meters between the two years - typically indicating data
    quality issues in one of the years - by dropping it from year 1.

    dataY1: pd.DataFrame
        Aggregated meter data for year 1. Must contain "component_id" and "n_meter".
    dataY2: pd.DataFrame
        Aggregated meter data for year 2. Must contain "component_id" and "n_meter".

    Returns
    -------
    pd.DataFrame
        Rows of `dataY1` whose "n_meter" matches the corresponding value in `dataY2`.
    """
    meter_counts = pd.merge(
        dataY1[["component_id", "n_meter"]].drop_duplicates(),
        dataY2[["component_id", "n_meter"]].drop_duplicates(),
        on="component_id",
        how="outer",
        suffixes=("Y1", "Y2"),
    ).reset_index(drop=True)

    meter_counts.fillna(0, inplace=True)
    meter_counts["keep_Y1"] = meter_counts["n_meterY1"] == meter_counts["n_meterY2"]

    return keep_BFEs(dataY1, meter_counts, "keep_Y1")

def get_aggregate_df(path_meter_year1: str, path_meter_save_year1: str, compID: pd.DataFrame, data_type: str,
                      path_meter_year2: str = None, path_meter_save_year2: str = None) -> None:
    """
    Aggregate meter, heat loss or annual energy data onto the user-defined
    "component_id" grouping, and save the result(s).

    path_meter_year1: str
        File path (.parquet) of the year-1 input data. For data_type="meter",
        output of B4_get_addProcessHM.py / cross_year_check(). For
        data_type="heatloss", output of B7_get_actualHeat.py. For
        data_type="annual", output of B8_get_energy.py.
    path_meter_save_year1: str
        File path (.parquet) where the aggregated year-1 result is saved.
    compID: pd.DataFrame
        User-specified component ID mapping table (see getCmpID4Agg()). Must
        contain "component_id", "serial" and "n_serial".
    data_type: str
        One of "meter", "heatloss" or "annual".
    path_meter_year2: str, optional
        File path (.parquet) of the year-2 input data. Required for
        data_type="meter" and data_type="annual" if a second year of data is
        available. Not used for data_type="heatloss" (heat loss predictions
        are not year-specific).
    path_meter_save_year2: str, optional
        File path (.parquet) where the aggregated year-2 result is saved.
        Required if `path_meter_year2` is provided.

    Returns
    -------
    None
        Writes the aggregated dataframe(s) to the provided file path(s).

    Notes
    -----
    For data_type="meter": if `path_meter_year2` is provided, components with
    a different number of meters between the two years are removed from year
    1 (see filter_unequal_meters()). If `path_meter_year2` is None, year 1 is
    aggregated and saved without this cross-year filter.

    For data_type="annual": if `path_meter_year2` is None, only year 1 is
    aggregated and saved.
    """
    if data_type == "meter":
        dataYR1 = wrang_proc_meter(data = pd.read_parquet(path_meter_year1, engine="pyarrow"),
                                   compID = compID, data_type="meter")

        if path_meter_year2 is not None:
            dataYR2 = wrang_proc_meter(data = pd.read_parquet(path_meter_year2, engine="pyarrow"),
                                       compID = compID, data_type="meter")

            dataYR1 = filter_unequal_meters(dataYR1, dataYR2)

            dataYR1.to_parquet(path_meter_save_year1)
            dataYR2.to_parquet(path_meter_save_year2)
        else:
            dataYR1.to_parquet(path_meter_save_year1)

    elif data_type == "heatloss":
        output = wrang_proc_meter(data = pd.read_parquet(path_meter_year1, engine="pyarrow"),
                                  compID = compID, data_type="heatloss")
        output.to_parquet(path_meter_save_year1)

    elif data_type == "annual":
        output = wrang_proc_meter(
            data = pd.read_parquet(path_meter_year1, engine="pyarrow"),
            compID = compID,
            data_type="annual")
        output.to_parquet(path_meter_save_year1)

        if path_meter_year2 is not None:
            output = wrang_proc_meter(
                data = pd.read_parquet(path_meter_year2, engine="pyarrow"),
                compID = compID,
                data_type="annual")
            output.to_parquet(path_meter_save_year2)
