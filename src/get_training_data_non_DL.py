"""

* Copyright (C) RWTH Aachen University, - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

path = Path(__file__).resolve().parents[1]

id_list = ["icustayid", "bloc", "max"]
demographics_list = ["gender", "age", "Weight_kg"]
vitals_list = ['GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1', 'mechvent', 'Shock_Index', 'SOFA',
               'SIRS']
lab_list_full = ['l_albumin', 'l_arterial_be', 'l_arterial_ph', 'l_bun', 'l_calcium', 'l_chloride', 'l_co2',
                 'l_creatinine',
                 'l_glucose', 'l_hb', 'l_hco3', 'l_inr', 'l_lactate', 'l_magnesium', 'l_paco2', 'l_pao2', 'l_pao2_fio2',
                 'l_platelets_count', 'l_potassium', 'l_pt', 'l_ptt', 'l_sodium', 'l_spo2', 'l_total_bili',
                 'l_wbccount']
lab_values_drop_list=['l_albumin', 'l_arterial_be', 'l_arterial_ph', 'l_bun', 'l_calcium', 'l_chloride',
                 'l_creatinine',
                 'l_glucose', 'l_hb', 'l_hco3', 'l_inr', 'l_lactate', 'l_magnesium', 'l_paco2', 'l_pao2', 'l_pao2_fio2',
                 'l_platelets_count', 'l_potassium', 'l_pt', 'l_ptt', 'l_sodium', 'l_spo2', 'l_total_bili',
                 'l_wbccount']
#lab_values_keep_list=['l_arterial_be', 'l_co2', 'l_hco3', 'l_paco2', 'l_pao2', 'l_platelets_count', 'l_potassium', 'l_sodium', 'l_spo2']
lab_values_keep_list = [item for item in lab_list_full if item not in lab_values_drop_list]
input_drop_list = id_list + demographics_list + vitals_list

output_drop_list = id_list


def get_training_data(window_length, testing_percent, lab_value_no):
    stacked_input_data_path = os.path.join(path, "data", "lab_values", "windowed_data_DVL",
                                           f"stacked_input_{window_length}.csv")
    stacked_output_data_path = os.path.join(path, "data", "lab_values", "windowed_data_DVL",
                                            f"stacked_output_{window_length}.csv")
    input_time_series_stacked = pd.read_csv(stacked_input_data_path)
    output_time_series_stacked = pd.read_csv(stacked_output_data_path)
    print(input_time_series_stacked.columns.tolist())
    input_time_series_stacked_edited = input_time_series_stacked.drop(input_drop_list, axis=1)
    output_time_series_stacked_edited = output_time_series_stacked.drop(output_drop_list, axis=1)
    output_columns_names = output_time_series_stacked_edited.columns.tolist()

    input_demographics = input_time_series_stacked[demographics_list].copy()
    input_vitals = input_time_series_stacked[vitals_list].copy()
    input_demographics_only_first = input_demographics.iloc[::window_length, :]
    '''
    print("input lab values : \n", input_time_series_stacked_edited)
    print("output lab values : \n", output_time_series_stacked_edited)
    print("input demographics : \n", input_demographics)
    print("input demographics last : \n", input_demographics_only_first)
    print("input vitals : \n", input_vitals)
    '''
    no_of_sequences = int(input_time_series_stacked_edited.shape[0] / window_length)
    print(f"number of sequences with length {window_length}= ", no_of_sequences)
    # reshape the input and output
    input_time_series_stacked_reshaped = input_time_series_stacked_edited.values.reshape(no_of_sequences, window_length,
                                                                                         input_time_series_stacked_edited.shape[
                                                                                             1])
    output_time_series_stacked_reshaped = output_time_series_stacked_edited.values.reshape(no_of_sequences,
                                                                                           output_time_series_stacked_edited.shape[
                                                                                               1])
    # Split the data
    x_l_train, x_l_test, y_train, y_test = train_test_split(
        input_time_series_stacked_reshaped,
        output_time_series_stacked_reshaped, test_size=testing_percent, shuffle=True, random_state=42)

    print("Train lab values shape", x_l_train.shape)
    print("Test lab values shape", x_l_test.shape)
    print("Train output lab values shape", y_train.shape)
    print("Test output lab values shape", y_test.shape)

    x_l_train = np.asarray(x_l_train).astype('float32')
    x_l_test = np.asarray(x_l_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    print("value of I ", lab_value_no)
    x_l_train_split = x_l_train[:, :, lab_value_no]
    x_l_test_split = x_l_test[:, :, lab_value_no]
    y_train_split = y_train[:, lab_value_no]
    y_test_split = y_test[:, lab_value_no]
    print(x_l_train_split.shape)
    print(x_l_test_split.shape)
    print(y_train_split.shape)
    print(y_test_split.shape)
    print(output_columns_names[lab_value_no])
    return x_l_train_split, x_l_test_split, y_train_split, y_test_split, output_columns_names[lab_value_no]


def get_training_data_eicu(window_length, testing_percent,lab_value_no):
    # read the input and output files
    stacked_input_data_path = os.path.join(path, "data", "lab_values", "windowed_data_eicu_DVL",
                                           f"stacked_input_{window_length}.csv")
    stacked_output_data_path = os.path.join(path, "data", "lab_values", "windowed_data_eicu_DVL",
                                            f"stacked_output_{window_length}.csv")

    input_time_series_stacked = pd.read_csv(stacked_input_data_path)
    output_time_series_stacked = pd.read_csv(stacked_output_data_path)

    print(input_time_series_stacked.columns.tolist())
    # print(output_time_series_stacked)

    input_time_series_stacked_edited = input_time_series_stacked.drop(input_drop_list, axis=1)
    output_time_series_stacked_edited = output_time_series_stacked.drop(output_drop_list, axis=1)
    output_columns_names = output_time_series_stacked_edited.columns.tolist()

    #print("input lab values eicu : \n", input_time_series_stacked_edited)
    #print("output lab values eicu : \n", output_time_series_stacked_edited)
    no_of_sequences = int(input_time_series_stacked_edited.shape[0] / window_length)

    # reshape the input and output
    input_time_series_stacked_reshaped = input_time_series_stacked_edited.values.reshape(no_of_sequences, window_length,
                                                                                         input_time_series_stacked_edited.shape[
                                                                                             1])
    output_time_series_stacked_reshaped = output_time_series_stacked_edited.values.reshape(no_of_sequences,
                                                                                           output_time_series_stacked_edited.shape[
                                                                                               1])

    # Split the data
    x_l_train, x_l_test, y_train, y_test = train_test_split(
        input_time_series_stacked_reshaped,
        output_time_series_stacked_reshaped, test_size=testing_percent, shuffle=True, random_state=42)
    #print("total number of sequences= ", no_of_sequences)
    #print("shape of input train lab values", x_l_train.shape)
    #print("shape of input test lab values", x_l_test.shape)
    #print("shape of train output values", y_train.shape)
    #print("shape of test output values", y_test.shape)

    print("value of I ", lab_value_no)
    x_l_train_split = x_l_train[:, :, lab_value_no]
    x_l_test_split = x_l_test[:, :, lab_value_no]
    y_train_split = y_train[:, lab_value_no]
    y_test_split = y_test[:, lab_value_no]
    print(x_l_train_split.shape)
    print(x_l_test_split.shape)
    print(y_train_split.shape)
    print(y_test_split.shape)
    return x_l_train_split, x_l_test_split, y_train_split, y_test_split,output_columns_names[lab_value_no]


if __name__ == '__main__':
    x_l_train_split, x_l_test_split, y_train_split, y_test_split, output_columns_name = get_training_data_eicu(
        window_length=5, testing_percent=0.1,lab_value_no=1)
