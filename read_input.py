import pandas as pd
import os

def read_index_data():
    directory = r'C:\Users\test\Desktop\SU\thesis\model\thesismodel\input'
    df = pd.read_excel(os.path.join(directory, 'index_data.xlsx'))

    return df

def read_riskfree_rate():
    # Read the CSV files into DataFrames
    df_1m = pd.read_csv('C:\\Users\\test\\Desktop\\SU\\thesis\\model\\thesismodel\\input\\United States 1-Month Bond Yield Historical Data.csv')
    df_1y = pd.read_csv('C:\\Users\\test\\Desktop\\SU\\thesis\\model\\thesismodel\\input\\United States 1-Year Bond Yield Historical Data.csv')
    df_10y = pd.read_csv('C:\\Users\\test\\Desktop\\SU\\thesis\\model\\thesismodel\\input\\United States 10-Year Bond Yield Historical Data.csv')
    # Merge the DataFrames on the 'Date' column
    merged_df = pd.merge(df_1m, df_1y, on='Date', suffixes=('_1m', '_1y'))
    merged_df = pd.merge(merged_df, df_10y, on='Date')

    # Select the 'Date' column and the 'Price' columns
    merged_df = merged_df[['Date', 'Price_1m', 'Price_1y', 'Price']]

    # Rename the columns for clarity
    merged_df.columns = ['Date', '1-Month Yield', '1-Year Yield', '10-Year Yield']

    merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    # Sort the DataFrame by the 'Date' column in ascending order
    merged_df = merged_df.sort_values(by='Date')

    # Reset index after sorting
    merged_df = merged_df.reset_index(drop=True)

    return merged_df