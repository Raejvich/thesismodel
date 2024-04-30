import pandas as pd
import os

def read_index_data():
    directory = r'C:\Users\test\Desktop\SU\thesis\model\thesismodel\input'
    df = pd.read_excel(os.path.join(directory, 'index_data.xlsx'))

    return df

