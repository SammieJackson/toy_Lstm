# import requests
import numpy as np
import pandas as pd
import xlrd
# from dateutil.relativedelta import relativedelta
# from datetime import datetime, timedelta, date

# Here we read the csv file, but since it contains data on ALL companies alltogether
# I did not figure out how to separate data for different comps from the csv (in excel you have sheets,
# but in the csv they seem to be unseparated)

# df = pd.read_csv("healthcare_data_new.csv")
# rows = df.shape[0]
# print(rows)

# Here we enlist all present companies and create an array of dataframes with data on each of them
companies = ['ABMD', 'ALK', 'AMGN', 'AXGN', 'BAX', 'BIO', 'BSTC', 'NHC', 'PDLI', 'TGTX', 'UTMD']
# ABMD_dat = pd.read_excel('healthcare_data_new.xls', sheet_name='ABMD')
all_data = []
for company in companies:
    tmp = pd.read_excel('healthcare_data_new.xls', sheet_name=company)
    all_data.append(tmp.values)


# We have 1304 entries (days). All tests should be of equal size. By now we're going to choose the number of tests 
# in which to break the whole series into by hand (as equal to 4). We would get us 4 smaller sequences 326 days each.
# Each of those we're going to split into input and output data according to the specified ratio
 
def split_data_for_single_company(company_data, number_of_samples=4, number_of_training_tests=1, in_to_out_ratio=3):
    total_entries = company_data.shape[0]
    if number_of_training_tests > number_of_samples:
        number_of_training_tests = max(1, number_of_samples - 1)
    number_of_validation_tests = number_of_samples - number_of_training_tests
    in_to_all_ratio = in_to_out_ratio / (in_to_out_ratio + 1)

    # we assume that it - rows in sample - would be an integer, since our samples have to be of equal size
    number_of_rows_in_sample = int( total_entries / number_of_samples )
    number_of_in_rows = int( number_of_rows_in_sample * in_to_all_ratio )
    number_of_out_rows = number_of_rows_in_sample - number_of_in_rows
    # print(number_of_rows_in_sample)
    # print(number_of_in_rows)
    # print(number_of_out_rows)

    train_inp = []
    train_out = []
    validate_inp = []
    validate_out = []

    for i in range(number_of_training_tests):
        low_index = i * number_of_rows_in_sample
        mid_index = low_index + number_of_in_rows # separates input and output
        top_index = low_index + number_of_rows_in_sample
        train_inp.append(np.matrix(company_data[low_index : mid_index]))
        train_out.append(np.matrix(company_data[mid_index : top_index]))

    for i in range(number_of_validation_tests):
        low_index = (i + number_of_training_tests) * number_of_rows_in_sample
        mid_index = low_index + number_of_in_rows # separates input and output
        top_index = low_index + number_of_rows_in_sample
        validate_inp.append(np.matrix(company_data[low_index : mid_index]))
        validate_out.append(np.matrix(company_data[mid_index : top_index]))

    return train_inp, train_out, validate_inp, validate_out

samples_for_one_company = all_data[0].shape[0]
# print(all_data[0][1:5])
abmd = np.matrix(all_data[0])
np.reshape(abmd, (samples_for_one_company, 16))
# print(all_data_0_np[:5])
ti, to, vi, vo = split_data_for_single_company(abmd, 4, 3, 3)
# print(len(vi[0]))
# print(len(vo[0]))
# print(vo[0][0])
print(type(vo[0][0]))
