import pandas as pd
import os

def read_index_data():
    directory = r'C:\Users\test\Desktop\SU\thesis\model\thesismodel\input'
    df = pd.read_excel(os.path.join(directory, 'index_data.xlsx'))

    return df

def read_riskfree_rate():
    directory = r'C:\Users\test\Desktop\SU\thesis\model\thesismodel\input'
    df = pd.read_excel(os.path.join(directory, 'Riskfree rate.xlsx'))
    df['Date'] = pd.to_datetime(df['Date'])

    return df