import pandas as pd
import read_input as r
from sklearn.preprocessing import MinMaxScaler


def sort_rf():
    """
    Sort from earliest date to latest

    """
    rf_raw = r.read_riskfree_rate()
    rf_raw['Date'] = pd.to_datetime(rf_raw['Date'])
    rf_sorted = rf_raw.sort_values(by='Date')
    rf_sorted = rf_sorted.reset_index(drop=True)
    rf_sorted = rf_sorted[['Date', 'Rate']]

    return rf_sorted


def match_data(rf, index_prices):
    """
    Find values in rf with corresponding date in index_price

    """
    merged_df = pd.merge(rf, index_prices, on='Date')

    return merged_df


def apply_scaler(df):
    """
    Apply min max scaler on index data
    """
    # Select columns to normalize
    columns_to_normalize = [
        'S&P 500', 	'S&P 500 Communication Services (Sector)',	
        'S&P 500 Consumer Discretionary (Sector)', 'S&P 500 Consumer Staples (Sector)',
        'S&P 500 Energy (Sector)', 'S&P 500 Financials (Sector)', 'S&P 500 Health Care (Sector)',
        'S&P 500 Industrials (Sector)',	'S&P 500 Information Technology (Sector)',	'S&P 500 Materials (Sector)',	
        'S&P 500 Real Estate (Sector)',	'S&P 500 Utilities (Sector)'
    ]

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns_to_normalize])

    # Convert the normalized data back to DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
    df_normalized = pd.concat([df.drop(columns=columns_to_normalize), normalized_df], axis=1)

    print(df_normalized)

    return df_normalized