import pandas as pd
import os
import pre_process as pp
import read_input as r
import model as m

def main():
    # Read risk free rate and index data
    rf = pp.sort_rf()
    index_prices = r.read_index_data()

    # Pick out data points
    data = pp.match_data(rf,index_prices)
    # get keys
    columns = data.columns.tolist()

    # Normalize prices
    data = pp.apply_scaler(data)    


    # first and second differences
    data = m.differences(data)
    print(data.head())

    # adf test
    m.adf_test(data, columns)



    return None

if __name__ == "__main__":
    main()