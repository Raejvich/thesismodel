import pandas as pd
import pre_process as pp
import read_input as r
import model as m
import copy

def main():
    # Read risk free rate and index data
    rf = r.read_riskfree_rate()
    print(rf.head())
    index_prices = r.read_index_data()

    # Pick out data points
    data = pp.match_data(rf,index_prices)
    # get keys
    columns = data.columns.tolist()

    # Normalize prices
    data = pp.apply_scaler(data)    

    data_differences = copy.deepcopy(data)
    # first and second differences
    data_differences = m.differences(data_differences)
    print(data_differences.head())

    # adf test
    adf_table = m.adf_test(data_differences, columns)
    print(adf_table)

    data = data.drop(columns=['Date'])
    result = m.create_var_model(data)



    return None

if __name__ == "__main__":
    main()