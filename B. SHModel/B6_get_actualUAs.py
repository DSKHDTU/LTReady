# -*- coding: utf-8 -*-
"""
B6_get_actualUAs.py

Trains the data-driven space heating (SH) model for each meter/building, to
obtain the "actual" heat transmission loss coefficients (UA values).

    The aim is to model the space heating load to obtain the actual heat transmission loss per building.
    We use a stochastic grey-box approach based on the energy balance equation and linear regression to determine the relevant physical properties of the building.

    The output variable: Space heating load in W per hour
    The input variable: Outdoor temperature, wind speed and sun per hour

    Model structure:

        Model is based on Rasmussen et al, 2020, Energies 15:

        HL = LSE[(UA0 + WsUAw)(Tb0 - Tout) - gAIg, Phi0] + e 
        
        Where LSE is the lnSumExp for smoothing function between the heat dependent and heat independent (Phi0) term.

        Model parameterized:
        HL(t) = ln[EXP[k*(UA0*(Tb0 - Tout(t)) + UAw*Ws(t)*(Tb0 - Tout(t)) - gA*Ig(t))] + EXP[Phi0*k]*1/k + e

        Due to overflow and underflow caused by large floating numbers in exp(float), we use the logExpSum trick which basically 
        results in the weather-dependent model (f).
        
        LSE(f,g) = 1/k * (m + ln(exp(k*f - m) + exp(kg - m))), where m = max(k*f, k*g)
                   
        The model objective function is

        minimize[ Sum((HL(t) - estimated_HL(t))^2) ]

        HL  - output, heat load
        Ws  - input, wind speed
        Tout - input, Outdoor temperature
        Ig - input, global solar irradiance

        UA0 - parameter, heat loss coefficient for transmission loss
        UAw - parameter, heat loss coefficient for ventilation and infiltration
        Tb0 - parameter, base temperature (outdoor temperature where building is in thermal balance, when HL = 0)
        gA - parameter, thermal solar fraction
        Phi0 - parameter, daily base heat load 
        k - hyperparameter for tuning the smoothing function

        e - noise term, N(0,sigma^2) i.i.d

    Model training and validation:

        We use all available dataset for training.
        We split the data into 80% training and 20% testing.

        Model validation:
            Parameter estimation statistics (p-value)
            Physical parameter calculation (estimated and confidence interval)
            Residual analysis (ACF, cummulated periodogram, histogram of residual, residual-variable scatter plot, SH and fitted.value across time, SH-Temperature scatter plot)
            Model accuracy, RN_RMSE
"""


import pandas as pd
import pickle

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t
import statsmodels.api as sm


def prep_data(df, selected_keys, split_ratio=0.8,
              on_cols=['address', 'location_id', 'serial', 'BFEnummer'],
              n_bins=10, random_state=None):
    """
    Prepare timeseries data for model training/testing with stratified random split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with identifiers, weather variables, and SH.
    selected_keys : dict
        Dictionary mapping identifier columns to their values, e.g.:
        {"address": "123 Main St", "location_id": "42", "serial": "A001", "BFEnummer": "1001"}.
    split_ratio : float
        Ratio for training/test split (default 0.8).
    on_cols : list of str
        Identifier columns (default: ['address', 'location_id', 'serial', 'BFEnummer']).
    n_bins : int
        Number of bins for stratification (default 10).
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    ts_train : pd.DataFrame
        Training dataset.
    ts_test : pd.DataFrame
        Test dataset.
    """

    # ------------------ Helper: Stratified split ------------------
    def stratified_split(values, split_ratio=0.8, n_bins=10, random_state=None):
        rng = np.random.default_rng(random_state)
        # Bin the values into n_bins
        bins = np.nanpercentile(values, np.linspace(0, 100, n_bins + 1))
        labels = np.digitize(values, bins[1:-1])  # returns bin labels 0..n_bins-1

        train_idx, test_idx = [], []
        for b in np.unique(labels):
            idx = np.where(labels == b)[0]
            rng.shuffle(idx)
            n_train = int(len(idx) * split_ratio)
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])


        return np.array(train_idx), np.array(test_idx)

    # ------------------ Prepare data ------------------
    # Ensure identifiers are strings
    for col in on_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Filter dataframe by selected identifiers
    mask = pd.Series(True, index=df.index)
    for col, val in selected_keys.items():
        mask &= df[col].astype(str) == str(val)
    ts_ready = df[mask].dropna().drop(columns=on_cols).reset_index()  # keep original index

    # Create interaction term
    if "Wind speed" in ts_ready.columns and "Temperature" in ts_ready.columns:
        ts_ready['Ws*Tout'] = ts_ready['Wind speed'] * ts_ready['Temperature']

    # Stratify by temperature to ensure cold/warm representation
    train_idx, test_idx = stratified_split(ts_ready['Temperature'].to_numpy(),
                                           split_ratio=split_ratio,
                                           n_bins=n_bins,
                                           random_state=random_state)

    ts_train = ts_ready.loc[train_idx].sort_values('date').reset_index(drop=True)
    ts_test = ts_ready.loc[test_idx].sort_values('date').reset_index(drop=True)

    return ts_train, ts_test

def model_solver(ts_train: pd.DataFrame, ts_test: pd.DataFrame, output_param: str = 'SH') -> tuple:
    """
    Fit the grey-box space heating model (see module docstring) to `ts_train`
    and evaluate it on `ts_test`.

    ts_train: pd.DataFrame
        Training data. Must contain columns "Temperature" (Tout), "Wind speed"
        (Ws), "Sun" (Ig) and `output_param` (default "SH").
    ts_test: pd.DataFrame
        Test data, with the same columns as `ts_train`.
    output_param: str, optional
        Name of the dependent variable column (default "SH").

    Returns
    -------
    tuple
        (estimated_params, confidence_intervals, R_squared_adj, RN_RMSE, RMSE,
        residuals, ts_test, optimizer_success, best_k):
            estimated_params: array of [UA0, UAw, Tb, gA, Phi0]
            confidence_intervals: list of (low, high) tuples per parameter
            R_squared_adj: adjusted R-squared on the test set
            RN_RMSE: range-normalized RMSE on the test set
            RMSE: RMSE on the test set
            residuals: test-set residuals
            ts_test: `ts_test` with an added "<output_param>.predict" column
            optimizer_success: bool, whether the final optimization succeeded
            best_k: the selected smoothing hyperparameter k

        If no optimization run succeeds, returns
        (None, None, None, None, None, None, ts_test, False, None).
    """
    # The model is based on Rasmussen et al, 2020, Energies 15 as described above
    # We use a nonlinear least square as objective function
    # We use a L-BFGS with boundary conditions to find local optimum for objective function (Rasmussen et al uses a global-local optimization approach without boundary condition)
    # We check if Hessian is convex (ensuring we have only global minimum)

    # We calculate the confidence interval for parameter estimation.
    # If confidence interval includes 0, we cannot reject the null_hypothesis, thus our parameter is not significant.
    # We remove the insignificant parameter, and run the model again with reduced parameters. The smallest model is UA0*(Tb - Ta)

    #--------------------- Set up model structure, objective functions and other relevant functions --------------------------------

    def model(params, k, Ws, Ta, Ig, use_Ws=True, use_Ig=True):
        
        # LogSumExp trick to stabilize for over- or underflow (source: Gregory Gundersen blo)
        def logsumexp_stable(f, g, k):
            kf, kg = f*k, g*k
            m = np.maximum(kf, kg)
            return (1.0/k) * (m + np.log(np.exp(kf - m) + np.exp(kg - m)))

        # Model formulation: we conditionally include the effect of Ws and Ig
        UA0, Tb, Phi0 = params[0], params[2], params[4]
        UAw = params[1] if use_Ws else 0
        gA = params[3] if use_Ig else 0

        f = UA0*(Tb - Ta) + UAw*Ws*(Tb - Ta) - gA*Ig
        g = Phi0

        return logsumexp_stable(f, g, k) 

    def residuals(params, k, Ws, Ta, Ig, HL, use_Ws=True, use_Ig=True):
        return HL - model(params, k, Ws, Ta, Ig, use_Ws, use_Ig)

    def objective(params, k, Ws, Ta, Ig, HL, use_Ws=True, use_Ig=True):
        return np.sum(residuals(params, k, Ws, Ta, Ig, HL, use_Ws, use_Ig)**2)

    def jacobian(params, k, Ws, Ta, Ig, use_Ws=True, use_Ig=True):
        
        """
        # Calculate Jacobian matrix
        # To calculate the variance of estimated parameters, we need the Hessian Matrix (2nd order derivate of obj func)
        # Since this can be computationally heavy to obtain we can approximate this value using the Jacobian Matrix.
        # The Jacobian Matrix is  defined as the derivate of the residual by the parameters.
        # It is used in estimating the covariance.        
        """
        
        UA0, UAw, Tb, gA, Phi0 = params       
        n = len(Ta)
        p = len(params)
        J = np.zeros((n, p))
        
        # compute f and softmax weight s safely
        if use_Ws:
            f = UA0*(Tb - Ta) + UAw*Ws*(Tb - Ta) - (gA*Ig if use_Ig else 0.0)
        else:
            f = UA0*(Tb - Ta) - (gA*Ig if use_Ig else 0.0)
    
        # s = exp(k f) / (exp(k f) + exp(k Phi0)) -> numerically stable logistic form
        s = 1.0 / (1.0 + np.exp(k*(Phi0 - f)))  # shape (n,)
    
        # precompute building blocks
        Tb_minus_Ta = (Tb - Ta)
        if use_Ws:
            UA_eff = UA0 + UAw*Ws
            Ws_Tb_minus_Ta = Ws * Tb_minus_Ta
        else:
            UA_eff = UA0
            Ws_Tb_minus_Ta = np.zeros_like(Ws)
    
        for i in range(n):
            si = s[i]
            # dr/dUA0
            J[i, 0] = - si * Tb_minus_Ta[i]
            # dr/dUAw
            J[i, 1] = - si * Ws_Tb_minus_Ta[i] if use_Ws else 0.0
            # dr/dTb
            J[i, 2] = - si * (UA_eff[i] if np.ndim(UA_eff) else UA_eff)
            # dr/dgA
            J[i, 3] = (si * Ig[i]) if use_Ig else 0.0
            # dr/dPhi0
            J[i, 4] = si - 1.0
    
        return J

    def conf_intv(hessian_matrix, params, k, Ws, Ta, Ig, HL, use_Ws=True, use_Ig=True, alpha = 0.05):
        # The covariance is given as the sigma^2 * invers(transpose(J)*J).
        # invers(transpose(J)*J) is the approximation of the Hessian matrix.

        # Calculate variance of residuals
        n = len(HL)
        p = len(params)
        sigma_squared = np.sum(residuals(params, k, Ws, Ta, Ig, HL, use_Ws, use_Ig)**2) / (n - p)

        try:
            # Covariance matrix
            cov_matrix = sigma_squared * np.linalg.inv(hessian_matrix)

        except np.linalg.LinAlgError:
            # Covariance matrix with psuedo-inverse
            cov_matrix = sigma_squared * np.linalg.pinv(hessian_matrix)

        # Standard errors for parameters - Ensure diagonal elements are non-negative before taking square root
        diag_variance = np.maximum(0, np.diag(cov_matrix))
        standard_errors = np.sqrt(diag_variance)

        # Confidence intervals (1 - alpha) - using the t-statistics
        t_critical = t.ppf(1 - alpha/2, n - p)
        confidence_intervals = [(param - t_critical * se, param + t_critical * se)
                                for param, se in zip(params, standard_errors)]

        return confidence_intervals

    def check_convex(hessian_matrix):
        try:
            # Check if optimization problem is convex (i.e. only a global minimum exist)
            # Check if Hessian is positive semi-definite
            eigenvalues = np.linalg.eigvals(hessian_matrix)
            is_convex = np.all(eigenvalues >= 0)

        except np.linalg.LinAlgError as e:
            print(f"Error in computing Hessian eigenvalues: {e}")
            is_convex = False

        print("Is the objective function convex?", is_convex) # If Singular then Hessian Matrix is Singular - Overparameterized, params > observation

        return is_convex
    
    def make_k_grid(HL, target = None, scaled = True):
        """
        Return k candidates tuned to the scale of 95th of HL - 5th HL (corresponding to the f - g terms).
        HL: array-like, measured heat load
        target: array of non-dimensional targets for k*scale (defaults to recommended values)
        scaled: activates or deactives scaling of target values
        """
        if target is None:
            target = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])   # try gentle → sharp
        else:
            target = np.array(target)
    
        if scaled:
            delta = HL - np.percentile(HL, 5)
            # robust scale: use 90th percentile of abs(delta) or IQR
            scale = np.percentile(np.abs(delta), 90.0)
            if scale <= 0:
                scale = max(np.std(delta), 1.0)  # fallback
        else:
            scale = 1
    
        k_candidates = target / scale
        
        return k_candidates

    def get_initials(Ta, HL):
        """
        Calculates initial values for parameter estimation to help the optimizer finding
        interpretable parameters

        Parameters
        ----------
        Ta : pd.Series()
            Outdoor temperature in celcius degree.
        HL : pd.Series()
            Daily average heat power in W.

        Returns
        -------
        list
            initial values for training models.

        """
        
        # Initial value for UA0 using simple OLS regression with intercept
        UA0 = sm.OLS(HL, sm.add_constant(Ta)).fit().params.iloc[1]
        
        # Correct sign - ensure that the sign is negative as it is expected, if not then set UA0 = 0
        if UA0 < 0:
            UA0 = abs(UA0)
        else:
            UA0 = 0
        
        # Initial value for Phi0 using 5th percentile value of HL
        Phi0 = np.percentile(HL, 5)
        
        # Initial value for Tb using average outdoor temperature
        Tb = np.mean(Ta)
                
        return [round(UA0,2), 0, round(Tb,2), 0, round(Phi0,2)]

    def run_model(Ws, Ta, Ig, HL, k, use_Ws=True, use_Ig=True):

        # Helper function to run the model and return results
        initial_guess = get_initials(Ta, HL) #[1, 0, min(Ta), 0, 0]  # Initial parameter guesses

        bounds = [(0, None),
                  (0, None) if use_Ws else (0, 0),
                  (-20, 25),
                  (0, None) if use_Ig else (0, 0), 
                  (0, None)]

        result = minimize(objective, initial_guess, args=(k, Ws, Ta, Ig, HL, use_Ws, use_Ig),
                          method='L-BFGS-B', bounds=bounds)
        return result

    def model_accuracy(y_true, y_pred, n_params):
        # Sample size and residuals
        resi = y_true - y_pred                                                                             # Calculate the residuals
        n = len(resi)                                                                                      # Number of observations
        k = n_params                                                                                       # Number of parameters

        # Calculate adjusted R-squared
        SS_res = np.sum(resi**2)                                                                           # Summed residual squared
        SS_tot = np.sum((y_true - np.mean(y_true))**2)                                                             # Total sum of squares
        R_squared_adj = 1 - (SS_res / SS_tot) * (n - 1) / (n - k - 1)

        # Calculate Range normalized root-mean square error - it is a robust version of the RMSE [-]
        RMSE = np.sqrt(1/n * SS_res)
        RN_RMSE = RMSE / (max(y_true) - min(y_true))

        RN_RMSE = round(RN_RMSE,4)
        RMSE = round(RMSE,4)
        
        return RN_RMSE, RMSE, R_squared_adj, resi

    #--------------------- Prepare training data --------------------------------
    # Train model and predict outcome using test data - set up dependent and independent variables
    HL = ts_train[output_param]

    Ta = ts_train['Temperature']
    Ws = ts_train['Wind speed']
    Ig = ts_train['Sun']

    k_candidates = make_k_grid(HL, target=[0.1,0.01,0.001,0.0001,0.00001,0.00001,0.000001,0.000001], scaled=False)

    # --------------------- Initial Full Model Run: Tuning hyperparameter k ----------------------------
    
    k_tuning_df = []
    
    for k in k_candidates:
        result = run_model(Ws, Ta, Ig, HL, k, use_Ws=True, use_Ig=True)
    
        if result.success:
            print("Optimization succeeded.")
            print('Total Evaluations:', result.nfev)
            
            # Save parameters
            estimated_params = result.x
            
            # Set up test data
            HL = ts_train[output_param]

            Ta = ts_train['Temperature']
            Ws = ts_train['Wind speed']
            Ig = ts_train['Sun']

            # Obtain prediction from test data
            output_param_predict = output_param + '.predict'
            HL_pred_train = model(estimated_params, k, Ws, Ta, Ig, use_Ws=True, use_Ig=True)
            
            # Save results
            k_tuning_df.append({'k': k, 
                                'RN_RMSE': model_accuracy(HL, HL_pred_train, len(estimated_params))[0],
                                'params': estimated_params
                                })
            
    if len(k_tuning_df) == 0:
        print("No successful optimization runs.")
        print("Optimization failed:", result.message)
        
        return None, None, None, None, None, None, ts_test, False, None
    
    else:
        print("Optimization successfull")
        
        k_tuning_df = pd.DataFrame(k_tuning_df)
        best_row = k_tuning_df.loc[k_tuning_df['RN_RMSE'].idxmin()]
        best_k = best_row['k']
        estimated_params = best_row['params']

    #--------------------- Model parameter uncertainty -----------------------------

    # Calculate Jacobian matrix
    J = jacobian(estimated_params, best_k, Ws, Ta, Ig, use_Ws=True, use_Ig=True)

    # Hessian matrix
    hessian_matrix = J.T @ J

    # Check if objective function is convex
    check_convex(hessian_matrix)

    # Calculate confidence interval
    confidence_intervals = conf_intv(hessian_matrix, estimated_params, best_k, Ws, Ta, Ig, HL)

    # Check significance of UAw and gA
    UAw_significant = confidence_intervals[1][0] > 0  # Check lower bound for UAw
    gA_significant = confidence_intervals[3][0] > 0   # Check lower bound for gA

    print(f"UAw significant: {UAw_significant}, gA significant: {gA_significant}")

    # --------------------- Re-run the model without insignificant parameters ----------------------------

    if not UAw_significant or not gA_significant:
        use_Ws = UAw_significant  # Only include Ws if UAw is significant
        use_Ig = gA_significant   # Only include Ig if gA is significant

        result = run_model(Ws, Ta, Ig, HL, best_k, use_Ws=use_Ws, use_Ig=use_Ig)

        if result.success:
            print("Reduced model - Optimization succeeded.")
            print('Reduced model - Total Evaluations:', result.nfev)

            # Save parameters
            estimated_params = result.x

            #--------------------- Model parameter uncertainty -----------------------------

            # Calculate Jacobian matrix
            J = jacobian(estimated_params, best_k, Ws, Ta, Ig, use_Ws=use_Ws, use_Ig=use_Ig)

            # Hessian matrix
            hessian_matrix = J.T @ J

            # Check if objective function is convex
            check_convex(hessian_matrix)

            # Calculate confidence interval
            confidence_intervals = conf_intv(hessian_matrix, estimated_params, best_k, Ws, Ta, Ig, HL, use_Ws=use_Ws, use_Ig=use_Ig)

    else:
        
        print("All parameters are significant; no need to reduce the model.")
        use_Ws = True
        use_Ig = True

    #--------------------- Model prediction using test data -----------------------------
    # Set up test data
    HL = ts_test[output_param]

    Ta = ts_test['Temperature']
    Ws = ts_test['Wind speed']
    Ig = ts_test['Sun']

    # Obtain prediction from test data
    output_param_predict = output_param + '.predict'
    ts_test[output_param_predict] = model(estimated_params, best_k, Ws, Ta, Ig, use_Ws=use_Ws, use_Ig=use_Ig)

    #-------------------------------- Model accuracy -------------------------------

    RN_RMSE, RMSE, R_squared_adj, resi = model_accuracy(HL, ts_test[output_param_predict], len(estimated_params))

    #-------------------------------------------------
    return estimated_params, confidence_intervals, R_squared_adj, RN_RMSE, RMSE, resi, ts_test, result.success, best_k

def run_modelling_scripts(path_load_data: str, path_save_data: str,
                          on_cols: list = ['address', 'location_id', 'serial', 'BFEnummer'],
                          seed: int = 42) -> None:
    """
    Train and test the space heating model for each unique building/meter
    defined by `on_cols`, and save the results.

    path_load_data: str
        File path (.pkl) of the daily timeseries dataset (output of
        B5_get_timeseries.py).
    path_save_data: str
        File path (.pkl) where the modelling results are saved.
    on_cols: list of str, optional
        Identifier columns for grouping (default: ['address', 'location_id', 'serial', 'BFEnummer']).
    seed: int, optional
        Random seed for the train/test split in prep_data(), ensuring
        reproducible model parameters across runs. Default 42. Set to
        None to disable (non-deterministic).

    Returns
    -------
    None
        Writes a dict to `path_save_data` containing, for each group:
        the identifier values, and a "Model_info" entry with the trained/test
        datasets, fitted parameters with confidence intervals, accuracy
        metrics (RN_RMSE, RMSE, R^2_adj), residuals, optimizer status and the
        selected hyperparameter k.
    """

    # Load timeseries data
    with open(path_load_data, 'rb') as f:
        timeseries = pickle.load(f)

    # Ensure identifiers are strings
    for col in on_cols:
        if col in timeseries.columns:
            timeseries[col] = timeseries[col].astype(str)

    # Get unique list of identifiers
    unique_keys = timeseries.reset_index()[on_cols].drop_duplicates().reset_index(drop=True)

    modeling_all = {col: [] for col in on_cols}
    modeling_all['Model_info'] = []

    # Iterate over each unique building/device
    for _, keys in unique_keys.iterrows():
        selected_keys = keys.to_dict()
        print("Processing:", selected_keys)

        # Prepare train/test datasets
        ts_train, ts_test = prep_data(timeseries, selected_keys, on_cols=on_cols,random_state=seed)

        # Train the model
        estimated_params, confidence_intervals, R_squared_adj, rn_rmse, rmse, residuals, ts_test, optimizer_status, k = model_solver(ts_train, ts_test)

        # Prepare params + confidence intervals
        estimated_params = pd.DataFrame([estimated_params], columns=['UA0','UAw','Tb','gA','Phi0']).T.rename(columns={0:'Coef'})
        conf = pd.DataFrame(confidence_intervals, columns=['Conf_low','Conf_up'], index=['UA0','UAw','Tb','gA','Phi0'])
        
        
        if not conf.empty:
            # If estimate = 0, set confidence interval to 0
            if estimated_params.loc['UAw','Coef'] == 0:
                conf.loc['UAw',['Conf_low', 'Conf_up']] = 0
            if estimated_params.loc['gA','Coef'] == 0:
                conf.loc['gA',['Conf_low', 'Conf_up']] = 0
            # If lower bound < 0 for UA0 and Phi0, set it to 0
            if conf.loc['UA0','Conf_low'] < 0:
                conf.loc['UA0','Conf_low'] = 0
            if conf.loc['Phi0','Conf_low'] < 0:
                conf.loc['Phi0','Conf_low'] = 0

        # Save model results
        modeling_bfe = {
            'ts_train': ts_train,
            'ts_test': ts_test,
            'params': pd.concat([estimated_params, conf], axis = 1),
            'RN_RMSE': rn_rmse,
            'RMSE': rmse,
            'R^2_adj': R_squared_adj,
            'residuals': residuals,
            'optimizer_success': optimizer_status,
            'hyperparam_k': k
        }

        # Add identifiers
        for col in on_cols:
            modeling_all[col].append(selected_keys[col])
        modeling_all['Model_info'].append(modeling_bfe)

    #Save results
    with open(path_save_data, 'wb') as f:     #WES
        pickle.dump(modeling_all, f)



