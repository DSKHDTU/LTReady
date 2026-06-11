# -*- coding: utf-8 -*-
"""
B7_get_actualHeat.py

Calculates the predicted space heating (SH) heat loss from the data-driven
"actual" UA values determined by B6_get_actualUAs.py, for a set of specified
outdoor conditions (indoor temperature, outdoor temperature, wind speed).
"""

import pickle
import pandas as pd
import numpy as np


def get_modelUAs(path_load_model: str, on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer']) -> pd.DataFrame:
    """
    Extract the fitted physical model parameters and error metrics from the
    saved model results, reshaping the central estimate ("Coef") and
    confidence bounds ("Conf_low", "Conf_up") into a single "index" column.

    path_load_model: str
        File path (.pkl) of the modelling results (output of
        B6_get_actualUAs.py, run_modelling_scripts()).
    on_cols: list of str, optional
        Identifier columns (default: ['address', 'location_id', 'serial', 'BFEnummer']).

    Returns
    -------
    pd.DataFrame
        One row per (identifier combination, "index" value), where "index"
        is one of "Coef", "Conf_low", "Conf_up". Columns include the model
        parameters (UA0, UAw, Tb, gA, Phi0), error metrics (RN_RMSE, RMSE)
        and the hyperparameter k.
    """
    # Load model results
    with open(path_load_model, 'rb') as f:
        modeling_all = pickle.load(f)

    records = []
    n_models = len(modeling_all['Model_info'])

    for i in range(n_models):
        model_info = modeling_all['Model_info'][i]['params']

        # If empty, create placeholder with NaNs
        if model_info.empty:
            model_info = pd.DataFrame(
                np.nan,
                index=pd.Index(['UA0','UAw','Tb','gA'], name="Parameter"),
                columns=['Coef','Conf_low','Conf_up']
            )

        # Reset index so Parameter becomes a column
        model_info = model_info.reset_index().rename(columns={'index': 'Parameter'})

        # Add identifier columns
        for col in on_cols:
            model_info[col] = modeling_all[col][i]

        # Add error metrics
        model_info['RN_RMSE'] = modeling_all['Model_info'][i]['RN_RMSE']
        model_info['RMSE'] = modeling_all['Model_info'][i]['RMSE']
        model_info['k'] = modeling_all['Model_info'][i]['hyperparam_k']

        # Apply RN_RMSE filtering
        mask = model_info['RN_RMSE'] >= 1
        model_info.loc[mask, ['Coef', 'Conf_low', 'Conf_up']] = np.nan

        # Melt Coef, Conf_low, Conf_up
        model_info_melted = model_info.melt(
            id_vars=['Parameter'] + on_cols + ['RN_RMSE','RMSE','k'],
            value_vars=['Coef','Conf_low','Conf_up'],
            var_name='index',
            value_name='value'
        )

        records.append(model_info_melted)

    # Concatenate all
    model_UAs = pd.concat(records, axis=0, ignore_index=True)

    # Pivot so 'Parameter' becomes columns
    model_UAs = model_UAs.pivot_table(
        index=['index'] + on_cols + ['RN_RMSE','RMSE','k'],
        columns='Parameter',
        values='value'
    ).reset_index()

    return model_UAs


def heatloss_calculation(UAs: pd.DataFrame, conditions: pd.DataFrame,
                          on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer'],
                          inclSun: bool = False, path_data_save: str = None,
                          onlySave: bool = True):
    """
    Calculate the predicted heat loss for each combination of model
    parameters (`UAs`) and outdoor condition (`conditions`), based on UA0
    (outdoor-temperature-dependent) and UAw (wind-speed-dependent)
    coefficients.

    UAs: pd.DataFrame
        Model parameters per identifier (output of get_modelUAs()), including
        columns "UA0", "UAw", "Tb", "Phi0", "k" and (if inclSun=True) "gA".
    conditions: pd.DataFrame
        Outdoor conditions to evaluate, with columns "Tin", "Tout" and "Ws"
        (and "Ig" if inclSun=True).
    on_cols: list of str, optional
        Identifier columns (default: ['address', 'location_id', 'serial', 'BFEnummer']).
    inclSun: bool, optional
        If True, include the solar gain term (gA * Ig) in the heat loss
        calculation. Requires `UAs` to contain "gA" and `conditions` to
        contain "Ig". Default False.
    path_data_save: str, optional
        File path (.parquet) where the result is saved, if `onlySave` is True.
    onlySave: bool, optional
        If True (default), save the result to `path_data_save` and return
        None. If False, return the result dataframe instead (and
        `path_data_save` is not required).

    Returns
    -------
    pd.DataFrame or None
        If `onlySave` is False, returns the dataframe (cartesian product of
        `UAs` and `conditions`) with an added "TotalHeatloss" column and
        "param_stats" identifying the parameter set ("Coef", "Conf_low" or
        "Conf_up"). If `onlySave` is True, returns None and writes the
        result to `path_data_save`.
    """
    # Cartesian product between UAs and conditions
    # Reset indices to avoid conflicts
    UAs = UAs.reset_index(drop=True)
    conditions = conditions.reset_index(drop=True)

    # Add cartesian key
    UAs['_key'] = 1
    conditions['_key'] = 1

    df = pd.merge(conditions, UAs, on='_key', how='outer').drop(columns=['_key'])

    # Heat loss calculations (vectorized)
    df['Heatloss_Tout'] = df['UA0'] * (df['Tb'] - df['Tout'])
    df['Heatloss_Ws*Tout'] = df['UAw'] * df['Ws'] * (df['Tb'] - df['Tout'])

    # Fill NaNs with 0
    df[['Heatloss_Tout', 'Heatloss_Ws*Tout']] = df[['Heatloss_Tout', 'Heatloss_Ws*Tout']].fillna(0)

    # Sum of transmission + wind losses
    f = df['Heatloss_Tout'] + df['Heatloss_Ws*Tout']

    if inclSun:
        df['Heatloss_gA*Ig'] = df['gA'] * df['Ig']
        f = f - df['Heatloss_gA*Ig']

    # Stabilizing term m
    m = np.maximum(f * df['k'], df['Phi0'] * df['k'])

    df['TotalHeatloss'] = (1.0 / df['k']) * (m + np.log(np.exp(f * df['k'] - m) + np.exp(df['Phi0'] * df['k'] - m)))

    df = df.reset_index(drop=True)

    df.rename(columns = {'index': 'param_stats'}, inplace = True)

    if onlySave:
        df.to_parquet(path_data_save)
    else:
        return df
