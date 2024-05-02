from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import pandas as pd

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
    
    print(adf_table)