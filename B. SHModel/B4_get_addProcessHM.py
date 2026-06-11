# -*- coding: utf-8 -*-
"""
B4_get_addProcessHM.py

Performs an additional cross-year quality check on the disaggregated space
heating (SH) data, when two years of meter data are available.

For each meter present in both years, an OLS model is fitted as
    heat_power = slope * outdoor_temperature + intercept
separately for each year. The slopes (and their confidence intervals) are
compared between years:
    - If the absolute difference between the two years' slopes is greater
      than 20%, AND their confidence intervals do not overlap, the
      year-to-year change is considered significant.
    - For meters with a significant change, the earlier year's data is
      excluded (since the more recent year is assumed to better represent
      the building's current state).

If only one year of meter data is provided, this cross-year check is skipped
and the single year's data is passed through unchanged.
"""

import pandas as pd

import statsmodels.formula.api as smf

def localize_to_copenhagen(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """
    Make a timestamp column timezone-aware in Europe/Copenhagen.

    df: pd.DataFrame
        Dataframe containing the timestamp column.
    col: str, optional
        Name of the timestamp column (default "timestamp").

    Returns
    -------
    pd.DataFrame
        `df` with `col` converted to Europe/Copenhagen timezone (localizing
        from UTC first if it was timezone-naive).
    """
    if df[col].dt.tz is None:
        # tz-naive → first localize to UTC, then convert
        df[col] = df[col].dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
    else:
        # already tz-aware → just convert
        df[col] = df[col].dt.tz_convert("Europe/Copenhagen")
    return df

def keep_meters(data: pd.DataFrame, df: pd.DataFrame, on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> pd.DataFrame:
    """
    Filter `data` so it only keeps rows where the combination of keys in
    `on_cols` exists in `df`.

    data: pd.DataFrame
        The main dataframe to filter.
    df: pd.DataFrame
        The dataframe with valid combinations of keys to keep.
    on_cols: list of str, optional
        Column names used for matching (default: ['address', 'location_id', 'serial', 'BFEnummer']).

    Returns
    -------
    pd.DataFrame
        Rows of `data` whose `on_cols` combination is present in `df`.
    """
    
    return data.merge(df[on_cols].drop_duplicates(), on=on_cols, how='inner')

def split_meter_data(dfY1: pd.DataFrame, dfY2: pd.DataFrame, on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> tuple:
    """
    Split meter identifiers into two groups based on whether they are present
    in both years of data or only one.

    dfY1: pd.DataFrame
        Year 1 meter data. Must contain a "year" column plus `on_cols`.
    dfY2: pd.DataFrame
        Year 2 meter data. Must contain a "year" column plus `on_cols`.
    on_cols: list of str, optional
        Identifier columns to match on (default: ['address', 'location_id', 'serial', 'BFEnummer']).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (common_df, missing_df): `common_df` lists the `on_cols` combinations
        present in both years; `missing_df` lists those present in only one year.
    """
    
    merged = (
        dfY1[on_cols + ['year']].drop_duplicates()
        .merge(dfY2[on_cols + ['year']].drop_duplicates(),
               on=on_cols, how='outer', suffixes=('_x', '_y'))
    )
    
    # Common meters (both years present)
    common_df = merged[merged['year_x'].notna() & merged['year_y'].notna()]
    
    # Missing meters (one year missing)
    missing_df = merged[(merged['year_x'].isna()) | (merged['year_y'].isna())]
    
    # Remove the redundant columns
    common_df.drop(columns=['year_x','year_y'], inplace = True)
    missing_df.drop(columns=['year_x','year_y'], inplace = True)
    
    return common_df, missing_df

def align_coltype(df: pd.DataFrame, on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> pd.DataFrame:
    """
    Ensure identifier columns are stored as strings, so that they merge
    consistently across the pipeline.

    df: pd.DataFrame
        Input dataframe.
    on_cols: list of str, optional
        Identifier columns to convert to string type (default:
        ['address', 'location_id', 'serial', 'BFEnummer']).

    Returns
    -------
    pd.DataFrame
        `df` with `on_cols` columns converted to string type (where present).
    """
    # Ensure identifiers are strings
    for col in on_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df

def year_to_year_deviation(df: pd.DataFrame,
                   on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer'],
                   year_col: str = "year",
                   y_col: str = "heat_power",
                   x_col: str = "Temperature",
                   alpha: float = 0.05,
                   full_output: bool = False):
    """
    Compare OLS slopes across two years for each group of meters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing meter readings, year, and grouping columns.
    on_cols : list of str
        Columns to group by.
    year_col : str
        Column indicating year.
    y_col : str
        Dependent variable.
    x_col : str
        Independent variable.
    alpha : float
        Significance level for confidence intervals.
    full_output : bool
        If True, return detailed results and Plotly figures (slopes & intercepts).
    
    Returns
    -------
    results : pd.DataFrame
    figs : dict of plotly.graph_objects.Figure (if full_output=True)
    """
    
    results = []
    slope_plot_data = []
    intercept_plot_data = []

    for keys, group in df.groupby(on_cols):
        if group[year_col].nunique() < 2:
            continue
        
        model = smf.ols(f"{y_col} ~ {x_col} * C({year_col})", data=group).fit()
        years = sorted(group[year_col].unique())
        year1, year2 = years[0], years[1]
        
        # Coefficients
        intercept_y1 = model.params['Intercept']
        slope_y1 = model.params[x_col]
        intercept_y2 = intercept_y1 + model.params.get(f"C({year_col})[T.{year2}]", 0)
        slope_y2 = slope_y1 + model.params.get(f"{x_col}:C({year_col})[T.{year2}]", 0)
        
        # Confidence intervals
        ci = model.conf_int(alpha=alpha)
        slope_y1_ci = ci.loc[x_col].tolist()
        if f"{x_col}:C({year_col})[T.{year2}]" in ci.index:
            slope_y2_ci = [
                slope_y1 + ci.loc[f"{x_col}:C({year_col})[T.{year2}]"][0],
                slope_y1 + ci.loc[f"{x_col}:C({year_col})[T.{year2}]"][1]
            ]
        else:
            slope_y2_ci = [float("nan"), float("nan")]
        
        # % difference in slopes
        def pct_diff(a, b):
            if a == 0:
                return float("nan")
            return abs((b - a) / a) * 100
        
        slope_diff_pct = round(pct_diff(slope_y1, slope_y2),1)
        
        # CI overlap check for slopes
        ci_overlap_slope = not (slope_y1_ci[1] < slope_y2_ci[0] or slope_y2_ci[1] < slope_y1_ci[0])
        slopes_different = (not ci_overlap_slope and slope_diff_pct >= 20)
        
        # Hypothesis test (joint F-test)
        ftest = model.f_test([f"C({year_col})[T.{year2}] = 0", 
                              f"{x_col}:C({year_col})[T.{year2}] = 0"])
        
        result = {
            **dict(zip(on_cols, keys)),
            "year1": year1,
            "year2": year2,
            "slope_y1": slope_y1,
            "slope_y2": slope_y2,
            "slope_diff_pct": slope_diff_pct,
            "slopes_different": slopes_different,
            "intercept_y1": intercept_y1,
            "intercept_y2": intercept_y2,
            "F_stat": float(ftest.fvalue),
            "p_value": float(ftest.pvalue),
        }
        
        if full_output:
            result.update({
                "slope_y1_CI_low": slope_y1_ci[0],
                "slope_y1_CI_high": slope_y1_ci[1],
                "slope_y2_CI_low": slope_y2_ci[0],
                "slope_y2_CI_high": slope_y2_ci[1],
                "ci_overlap_slope": ci_overlap_slope,
            })
        
        results.append(result)
        
        # Collect plotting data if needed
        if full_output:
            group_id = "_".join([str(k) for k in keys])
            slope_plot_data += [
                {"group": group_id, "year": year1, "value": slope_y1, "ci_low": slope_y1_ci[0], "ci_high": slope_y1_ci[1]},
                {"group": group_id, "year": year2, "value": slope_y2, "ci_low": slope_y2_ci[0], "ci_high": slope_y2_ci[1]},
            ]
            intercept_plot_data += [
                {"group": group_id, "year": year1, "value": intercept_y1},
                {"group": group_id, "year": year2, "value": intercept_y2},
            ]
    
    results_df = pd.DataFrame(results)
    
    if full_output:
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.io as pio
        pio.renderers.default='browser'
        
        figs = {}
        
        # Slope plot
        if slope_plot_data:
            slope_df = pd.DataFrame(slope_plot_data)
            fig_slope = go.Figure()
            for group, gdf in slope_df.groupby("group"):
                fig_slope.add_trace(go.Scatter(
                    x=gdf["year"], y=gdf["value"],
                    mode="markers",
                    name=f"{group} slopes",
                    error_y=dict(
                        type="data", symmetric=False,
                        array=gdf["ci_high"] - gdf["value"],
                        arrayminus=gdf["value"] - gdf["ci_low"]
                    )
                ))
            fig_slope.update_layout(title=f"Slopes with 95% CI. Slope mean abs diff: {slope_diff_pct}%", 
                                    xaxis_title="Year", yaxis_title="Slope", 
                                    template="plotly_white")
            figs["slopes"] = fig_slope
        
        # Intercept plot (mean values only)
        if intercept_plot_data:
            intercept_df = pd.DataFrame(intercept_plot_data)
            fig_int = go.Figure()
            for group, gdf in intercept_df.groupby("group"):
                fig_int.add_trace(go.Scatter(
                    x=gdf["year"], y=gdf["value"],
                    mode="markers+lines",
                    name=f"{group} intercepts"
                ))
            fig_int.update_layout(title="Intercept Means", 
                                  xaxis_title="Year", yaxis_title="Intercept", 
                                  template="plotly_white")
            figs["intercepts"] = fig_int
        
        return results_df, figs
    
    return results_df

def cross_year_check(path_SH_year1: str, path_weather_year1: str, path_save_meter_year1: str, YEAR1: str,
                      path_SH_year2: str = None, path_weather_year2: str = None,
                      path_save_meter_year2: str = None, YEAR2: str = None) -> None:
    """
    Compare year-1 and year-2 space heating data per meter and remove meters
    whose heat-power-vs-temperature relationship changed significantly
    between the two years (see module docstring for the test applied).

    If only year-1 data is provided (path_SH_year2=None), no cross-year
    comparison is performed and the year-1 data is passed through unchanged
    (after timezone localization).

    path_SH_year1: str
        File path (.parquet) of the disaggregated SH data for year 1 (output
        of B2_get_disaggregated.py). Must contain "serial", "timestamp" and
        "heat_power".
    path_weather_year1: str
        File path (.parquet) of the hourly weather data for year 1 (output of
        B3_get_weather_data.py).
    path_save_meter_year1: str
        File path (.parquet) where the (possibly filtered) year-1 data is saved.
    YEAR1: str
        Calendar year of `path_SH_year1`, as a string (e.g. "2022").
    path_SH_year2: str, optional
        File path (.parquet) of the disaggregated SH data for year 2. If
        None (default), the cross-year check is skipped.
    path_weather_year2: str, optional
        File path (.parquet) of the hourly weather data for year 2. Required
        if `path_SH_year2` is provided.
    path_save_meter_year2: str, optional
        File path (.parquet) where the year-2 data is saved. Required if
        `path_SH_year2` is provided.
    YEAR2: str, optional
        Calendar year of `path_SH_year2`, as a string (e.g. "2023"). Required
        if `path_SH_year2` is provided.

    Returns
    -------
    None
        Writes the (possibly filtered) year-1 data to `path_save_meter_year1`,
        and (if year 2 is provided) the year-2 data to `path_save_meter_year2`.
    """
    dataYR1 = pd.read_parquet(path_SH_year1, engine='pyarrow')
    weatherY1 = pd.read_parquet(path_weather_year1, engine='pyarrow')

    # Align column type
    dataYR1 = align_coltype(dataYR1, on_cols=['serial'])

    # Set timezone to Copenhagen time for merging dataframes
    dataYR1 = localize_to_copenhagen(dataYR1, "timestamp")
    weatherY1 = localize_to_copenhagen(weatherY1, "timestamp")

    # Merge meter and weather data
    dataYR1 = pd.merge(dataYR1, weatherY1)

    # Add column year
    dataYR1['year'] = dataYR1['timestamp'].dt.year.astype(int).astype(str)

    # Remove timestamps that don't belong to the meter data's stated year
    dataYR1.query('year==@YEAR1', inplace = True)

    if path_SH_year2 is None:
        # Only one year of meter data is available - skip the cross-year check
        dataYR1.to_parquet(path_save_meter_year1)
        return

    dataYR2 = pd.read_parquet(path_SH_year2, engine='pyarrow')
    weatherY2 = pd.read_parquet(path_weather_year2, engine='pyarrow')

    dataYR2 = align_coltype(dataYR2, on_cols=['serial'])

    dataYR2 = localize_to_copenhagen(dataYR2, "timestamp")
    weatherY2 = localize_to_copenhagen(weatherY2, "timestamp")

    dataYR2 = pd.merge(dataYR2, weatherY2)

    dataYR2['year'] = dataYR2['timestamp'].dt.year.astype(int).astype(str)

    dataYR2.query('year==@YEAR2', inplace = True)

    # Continue filtration only for the meters in which data is present for both years
    common_df, missing_df = split_meter_data(dataYR1, dataYR2, on_cols=['serial'])

    # Only look at colder weather & Year-to-year-deviation filtration is only performed on meters containing data for both years
    dYR2 = keep_meters(dataYR2.query('Temperature<=15'), common_df, on_cols=['serial'])
    dYR1 = keep_meters(dataYR1.query('Temperature<=15'), common_df, on_cols=['serial'])

    # Merge the years together
    df = pd.concat([dYR1,dYR2]).reset_index(drop=True)

    # Run the script to get the meters with no year-to-year deviation
    devdf = year_to_year_deviation(df, on_cols=['serial'])
    devdf = devdf.query('slopes_different==False')[['serial']].drop_duplicates()

    # The list of IDs to keep
    IDs_to_keep = pd.concat([missing_df, devdf])

    # Keep the meters with no significant deviation
    dataYR1 = keep_meters(dataYR1, IDs_to_keep, on_cols=['serial'])

    dataYR1.to_parquet(path_save_meter_year1)
    dataYR2.to_parquet(path_save_meter_year2)
