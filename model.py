from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def differences(data):
    """
    calculate first and second differences
    """
    for index in data.drop(columns=['Date']):
        data[f'{index}_dif'] = data[f'{index}']-data[f'{index}'].shift(1)
        data[f'{index}_dif2'] = data[f'{index}_dif']-data[f'{index}_dif'].shift(1)
    data.dropna(inplace=True)
    
    return data


def adf_test(data, tickers):
    """
    ADF test to determine which difference to use for stationary data
    """
    adf_table = pd.DataFrame(index=tickers, columns=['prices_pvalue','dif_pvalue', 'dif2_pvalue','Integration_order'])

    # Apply the ADF test to the three time series of each asset
    for ticker in tickers:
        if ticker != "Date":
            adf_table.loc[ticker,'prices_pvalue'] = adfuller(data[f'{ticker}'],autolag='aic',regression='c')[1]
            adf_table.loc[ticker,'dif_pvalue'] = adfuller(data[f'{ticker}_dif'],autolag='aic',regression='n')[1]
            adf_table.loc[ticker,'dif2_pvalue'] = adfuller(data[f'{ticker}_dif2'],autolag='aic',regression='n')[1]
        
    # We identify the order of integration of each asset and save it in the respective column
    for ticker in tickers:
        if adf_table.loc[ticker,'prices_pvalue']<0.05:
            adf_table.loc[ticker,'Integration_order'] = 0
        elif adf_table.loc[ticker,'dif_pvalue']<0.05:
            adf_table.loc[ticker,'Integration_order'] = 1
        elif adf_table.loc[ticker,'dif2_pvalue']<0.05:
            adf_table.loc[ticker,'Integration_order'] = 2
    adf_table.dropna(inplace=True)
    
    return adf_table


def create_var_model(data):

    indexes = [
    "S&P 500",
    "S&P 500 Communication Services (Sector)",
    "S&P 500 Consumer Discretionary (Sector)",
    "S&P 500 Consumer Staples (Sector)",
    "S&P 500 Energy (Sector)",
    "S&P 500 Financials (Sector)",
    "S&P 500 Health Care (Sector)",
    "S&P 500 Industrials (Sector)",
    "S&P 500 Information Technology (Sector)",
    "S&P 500 Materials (Sector)",
    "S&P 500 Real Estate (Sector)",
    "S&P 500 Utilities (Sector)"
    ]

    for index_name in indexes:
        print("Fitting VAR model for", index_name)
        # Extract data for the current index
        index_data = data[[index_name, '1-Month Yield', '1-Year Yield', '10-Year Yield']]
        
        # Drop missing values
        index_data = index_data.dropna()
        
        # Create VAR model object
        model = VAR(index_data)
        
        # Fit VAR model
        results = model.fit(1)
        
        # Print estimation summary
        print(results.summary())


def granger_causality_for_indexes1(data, maxlag=1):
    """
    Perform Granger causality tests for each index in the list.
    
    Parameters:
        data (DataFrame): DataFrame containing the data for the indexes and other variables.
        indexes (list): List of index names.
        maxlag (int): Maximum lag order for the Granger causality tests.
        
    Returns:
        dict: Dictionary containing the Granger causality test results for each index.
    """
    granger_results = {}

    indexes = [
    "S&P 500",
    "S&P 500 Communication Services (Sector)",
    "S&P 500 Consumer Discretionary (Sector)",
    "S&P 500 Consumer Staples (Sector)",
    "S&P 500 Energy (Sector)",
    "S&P 500 Financials (Sector)",
    "S&P 500 Health Care (Sector)",
    "S&P 500 Industrials (Sector)",
    "S&P 500 Information Technology (Sector)",
    "S&P 500 Materials (Sector)",
    "S&P 500 Real Estate (Sector)",
    "S&P 500 Utilities (Sector)"
    ]
    
    for index_name in indexes:
        print("Performing Granger causality test for", index_name)
        
        # Extract data for the current index
        index_data = data[[index_name, '1-Month Yield']]
        
        # Drop missing values
        index_data = index_data.dropna()
        # Perform Granger causality test
        granger_test_result = grangercausalitytests(index_data, maxlag, addconst=True, verbose=None)
        
        # Store the results in the dictionary
        granger_results[index_name] = granger_test_result

    
        # Print the results
    for index_name, results in granger_results.items():
        print("1-month yield Granger causality test results for", index_name)
        for lag, result in results.items():
            print("Lag:", lag)
            for test_statistic, p_value in result[0].items():
                print(f"{test_statistic}: p-value =", p_value)
        
    return granger_results


def granger_causality_for_indexes2(data, maxlag=1):
    """
    Perform Granger causality tests for each index in the list.
    
    Parameters:
        data (DataFrame): DataFrame containing the data for the indexes and other variables.
        indexes (list): List of index names.
        maxlag (int): Maximum lag order for the Granger causality tests.
        
    Returns:
        dict: Dictionary containing the Granger causality test results for each index.
    """
    granger_results = {}

    indexes = [
    "S&P 500",
    "S&P 500 Communication Services (Sector)",
    "S&P 500 Consumer Discretionary (Sector)",
    "S&P 500 Consumer Staples (Sector)",
    "S&P 500 Energy (Sector)",
    "S&P 500 Financials (Sector)",
    "S&P 500 Health Care (Sector)",
    "S&P 500 Industrials (Sector)",
    "S&P 500 Information Technology (Sector)",
    "S&P 500 Materials (Sector)",
    "S&P 500 Real Estate (Sector)",
    "S&P 500 Utilities (Sector)"
    ]
    
    for index_name in indexes:
        print("Performing Granger causality test for", index_name)
        
        # Extract data for the current index
        index_data = data[[index_name, '1-Year Yield']]
        
        # Drop missing values
        index_data = index_data.dropna()
        # Perform Granger causality test
        granger_test_result = grangercausalitytests(index_data, maxlag, addconst=True, verbose=None)
        
        # Store the results in the dictionary
        granger_results[index_name] = granger_test_result

    
        # Print the results
    for index_name, results in granger_results.items():
        print("1-year yield Granger causality test results for", index_name)
        for lag, result in results.items():
            print("Lag:", lag)
            for test_statistic, p_value in result[0].items():
                print(f"{test_statistic}: p-value =", p_value)
        
    return granger_results


def granger_causality_for_indexes3(data, maxlag=1):
    """
    Perform Granger causality tests for each index in the list.
    
    Parameters:
        data (DataFrame): DataFrame containing the data for the indexes and other variables.
        indexes (list): List of index names.
        maxlag (int): Maximum lag order for the Granger causality tests.
        
    Returns:
        dict: Dictionary containing the Granger causality test results for each index.
    """
    granger_results = {}

    indexes = [
    "S&P 500",
    "S&P 500 Communication Services (Sector)",
    "S&P 500 Consumer Discretionary (Sector)",
    "S&P 500 Consumer Staples (Sector)",
    "S&P 500 Energy (Sector)",
    "S&P 500 Financials (Sector)",
    "S&P 500 Health Care (Sector)",
    "S&P 500 Industrials (Sector)",
    "S&P 500 Information Technology (Sector)",
    "S&P 500 Materials (Sector)",
    "S&P 500 Real Estate (Sector)",
    "S&P 500 Utilities (Sector)"
    ]
    
    for index_name in indexes:
        print("Performing Granger causality test for", index_name)
        
        # Extract data for the current index
        index_data = data[[index_name, '10-Year Yield']]
        
        # Drop missing values
        index_data = index_data.dropna()
        # Perform Granger causality test
        granger_test_result = grangercausalitytests(index_data, maxlag, addconst=True, verbose=None)
        
        # Store the results in the dictionary
        granger_results[index_name] = granger_test_result

    
        # Print the results
    for index_name, results in granger_results.items():
        print("10-year yield Granger causality test results for", index_name)
        for lag, result in results.items():
            print("Lag:", lag)
            for test_statistic, p_value in result[0].items():
                print(f"{test_statistic}: p-value =", p_value)
        
    return granger_results