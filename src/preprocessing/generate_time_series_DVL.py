"""

* Copyright (C) RWTH Aachen University, - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""

import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import logging

# set basic configs:
logging.basicConfig(level=logging.DEBUG)
cwd = os.path.dirname(os.path.abspath(__file__))

# input and output files loaded:
input_data_path = os.path.join(cwd, "..", '..', "data", "lab_values", "input_w_lab_rearranged.csv")
output_data_path = os.path.join(cwd, "..", '..', "data", "lab_values", "lab_results.csv")

# list of features in demographics
demo_cols = ["gender", "age", "Weight_kg"]

# list of features in vitals
vitals_cols = ['GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1', 'mechvent', 'Shock_Index', 'SOFA',
               'SIRS']
# list of columns to drop from input data
drop_list_input = ["icustayid", "bloc", "max"] + demo_cols + vitals_cols
drop_list_output = ["icustayid", "bloc", "max"]


# if you want to drop more lab values, uncomment the next.
# = ["icustayid", "bloc", "max","l_ptt", "l_pt", "l_potassium", "l_pao2_fio2", "l_magnesium"]


def scale_data(data):
    data_ids = data.iloc[:, :2]
    data_no = data.iloc[:, 2:].copy()
    scaler = MinMaxScaler()
    scaled_ds = scaler.fit_transform(data_no)
    # save the scaler
    dump(scaler, open(os.path.join(cwd, 'scaler_mimic.pkl'), 'wb'))
    scaled_ds = np.round(scaled_ds, decimals=2)
    scaled_ds = pd.DataFrame(scaled_ds, columns=data_no.columns)
    new_df = pd.concat([data_ids, scaled_ds], axis=1)
    return new_df


def read_data_mimic(window_length, min_window_length=4, max_window_length=20):
    # read the input and output files
    input_data = pd.read_csv(input_data_path, )
    output_data = pd.read_csv(output_data_path, )
    # scale the input data
    input_data_scaled = scale_data(input_data)
    # filter the dataset to include stays with more than min_window_length timestamps (min_window_length* 4 hours )
    input_data_scaled = input_data_scaled.groupby('icustayid').filter(lambda x: len(x) > min_window_length)
    output_data = output_data.groupby('icustayid').filter(lambda x: len(x) > min_window_length)
    # filter the dataset to include stays with less than max_window_length timestamps (max_window_length* 4 hours )
    input_data_scaled = input_data_scaled.groupby('icustayid').filter(lambda x: len(x) < max_window_length)
    output_data = output_data.groupby('icustayid').filter(lambda x: len(x) < max_window_length)
    # get the number of unique icustays
    y = input_data_scaled.groupby('icustayid').ngroups
    logging.debug(f"number of unique icu stays from MIMIC dataset== {y}")
    # get the max length for each icustay and show the frequency of each length
    seq_last_value = output_data.groupby("icustayid").tail(1).reset_index(drop=True)
    seq_last_value["bloc"] = seq_last_value["bloc"]
    max_Seq_length = seq_last_value["bloc"].max() + 1
    logging.debug(f"max sequence length= {max_Seq_length - 1}")
    logging.debug(f"sequence length and count: \n {seq_last_value['bloc'].value_counts()}")
    # calculate the length of each input sequence and sort the sequences according to length
    input_data_scaled["max"] = input_data_scaled.groupby(['icustayid'])['bloc'].transform('max')
    input_data_scaled = input_data_scaled.sort_values(['max', 'icustayid', 'bloc'], ascending=True)
    output_data["max"] = output_data.groupby(['icustayid'])['bloc'].transform('max')
    output_data = output_data.sort_values(['max', 'icustayid', 'bloc'], ascending=True)
    # create two dataframes to save the sub-sequences of equal window lengths
    input_time_series_stacked = pd.DataFrame(columns=input_data_scaled.columns.tolist())
    output_time_series_stacked = pd.DataFrame(columns=output_data.columns.tolist())
    logging.debug(f"Input data scaled :\n {input_data_scaled}")

    for i in range(window_length + 1, max_Seq_length):
        sub_x = input_data_scaled.loc[input_data_scaled["max"] == i]
        sub_y = output_data.loc[output_data["max"] == i]
        # print(f"X sequences with length {i} \n", sub_x)
        # print(f"Y sequences with length {i} \n", sub_y)

        for window_shift in range(window_length, i):
            if window_shift == window_length:
                grouped_x = sub_x.groupby(['icustayid'])
                windowed_x = grouped_x.head(window_length)
                grouped_y = sub_y.groupby(['icustayid'])
                windowed_y_all = grouped_y.head(window_length + 1)
                windowed_y_one = windowed_y_all.groupby(['icustayid']).tail(1)
                # print("Full X window \n", windowed_x)
                # print("Full Y window \n", windowed_y_one)
                input_time_series_stacked = input_time_series_stacked.append(windowed_x, ignore_index=True)
                output_time_series_stacked = output_time_series_stacked.append(windowed_y_one, ignore_index=True)
            else:
                grouped_x = sub_x.groupby(['icustayid'])
                windowed_x = grouped_x.head(window_shift)
                grouped_y = sub_y.groupby(['icustayid'])
                windowed_y_all = grouped_y.head(window_shift + 1)
                no_of_rows_to_drop = window_shift - window_length
                grouped_x_sub = windowed_x.groupby(['icustayid'])
                rows_to_delete = grouped_x_sub.head(no_of_rows_to_drop)
                rows_to_delete_x_index = rows_to_delete.index.tolist()
                windowed_y_one = windowed_y_all.groupby(['icustayid']).tail(1)
                windowed_x.drop(rows_to_delete_x_index, axis=0, inplace=True)
                input_time_series_stacked = input_time_series_stacked.append(windowed_x, ignore_index=True)
                output_time_series_stacked = output_time_series_stacked.append(windowed_y_one, ignore_index=True)

    # Save the raw stacked windowed input/output data with icu stay id and bloc. Uncomment to execute
    '''
    stacked_input_data_path = os.path.join(cwd, "..", '..', "data", "lab_values", "windowed_data_DVL",
                                           f"stacked_input_{window_length}.csv")
    stacked_output_data_path = os.path.join(cwd, "..", '..', "data", "lab_values", "windowed_data_DVL",
                                            f"stacked_output_{window_length}.csv")
    input_time_series_stacked.to_csv(stacked_input_data_path, index=False)
    output_time_series_stacked.to_csv(stacked_output_data_path, index=False)
    '''

    # drop columns from input and output data
    input_time_series_stacked_edited = input_time_series_stacked.drop(drop_list_input, axis=1)
    output_time_series_stacked_edited = output_time_series_stacked.drop(drop_list_output, axis=1)
    # extract vitals and demographics from input data
    input_demographics = input_time_series_stacked[demo_cols].copy()
    input_vitals = input_time_series_stacked[vitals_cols].copy()
    # for demographics we take only the first row from each icu_stay
    input_demographics_only_first = input_demographics.iloc[::window_length, :]
    logging.debug(f"stacked input lab values :\n {input_time_series_stacked_edited}")
    logging.debug(f"stacked output lab values :\n {output_time_series_stacked_edited}")
    logging.debug(f"Input demographics last :\n {input_demographics_only_first}")
    logging.debug(f"Input vitals:\n {input_vitals}")

    no_of_sequences = int(input_time_series_stacked_edited.shape[0] / window_length)

    logging.debug(f"Total number of sequences from MIMIC with length= {window_length} equals to {no_of_sequences} \n")

    return input_time_series_stacked_edited, input_vitals, input_demographics_only_first, output_time_series_stacked_edited


if __name__ == "__main__":
    # testing_percent=0.1
    # validation_percent = 0.1
    min_window_length = 4
    max_window_length = 20
    window_length = 6

    read_data_mimic(window_length, min_window_length, max_window_length)
