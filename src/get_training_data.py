"""

* Copyright (C) RWTH Aachen University, - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# imports from directory
from src.preprocessing.generate_time_series_DVL import read_data_mimic
from src.preprocessing.generate_time_series_DVL_eicu import read_data_eicu

# set basic configs for logging and paths:
logging.basicConfig(level=logging.DEBUG)
path = Path(__file__).resolve().parents[1]



def get_training_data_mimic(window_length=5, testing_percent=0.1):
    input_lab, input_vitals, input_demo, output = read_data_mimic(window_length)


    # get no of features for input, output, demographics, vitals
    no_features_lab = input_lab.shape[1]
    no_features_vitals = input_vitals.shape[1]
    no_features_demo = input_demo.shape[1]
    no_output = output.shape[1]
    logging.debug(f"no of lab features= {no_features_lab} \n")
    logging.debug(f"no of vitals features= {no_features_vitals} \n")
    logging.debug(f"no of demographics features= {no_features_demo} \n")
    logging.debug(f"no of output features= {no_output} \n")
    # get the total number of sequences used for the model training
    no_of_sequences = int(input_lab.shape[0]/window_length)
    logging.debug(f"Total number of sequences with length= {window_length} equals to {no_of_sequences} \n")
    # get the list of the output lab values' names
    output_columns_names = output.columns.tolist()
    # reshape the input and output
    input_lab_reshaped = input_lab.values.reshape(no_of_sequences, window_length, no_features_lab)
    input_vitals_reshaped = input_vitals.values.reshape(no_of_sequences, window_length, no_features_vitals)
    input_demo_reshaped = input_demo.values.reshape(no_of_sequences, no_features_demo)
    output_reshaped = output.values.reshape(no_of_sequences, no_output)
    # Split the data
    x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test = train_test_split(
        input_lab_reshaped, input_vitals_reshaped, input_demo_reshaped,
        output_reshaped, test_size=testing_percent, shuffle=True, random_state=42)

    logging.debug(f"Train lab values shape = {x_l_train.shape}")
    logging.debug(f"Test lab values shape = {x_l_test.shape}")
    logging.debug(f"Train lab vitals shape = {x_v_train.shape}")
    logging.debug(f"Test lab vitals shape = {x_v_test.shape}")
    logging.debug(f"Train demographics shape = {x_d_train.shape}")
    logging.debug(f"Test demographics shape = {x_d_test.shape}")
    logging.debug(f"Train output lab values shape = {y_train.shape}")
    logging.debug(f"Test output lab values shape = {y_test.shape}")
    # set the types for the numpy arrays to avoid compatibility issues
    x_l_train = np.asarray(x_l_train).astype('float32')
    x_l_test = np.asarray(x_l_test).astype('float32')
    x_v_train = np.asarray(x_v_train).astype('float32')
    x_v_test = np.asarray(x_v_test).astype('float32')
    x_d_train = np.asarray(x_d_train).astype('float32')
    x_d_test = np.asarray(x_d_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    return x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names


def get_training_data_eicu(window_length, testing_percent):
    input_lab, input_vitals, input_demo, output = read_data_eicu(window_length)

    # get no of features for input, output, demographics, vitals
    no_features_lab = input_lab.shape[1]
    no_features_vitals = input_vitals.shape[1]
    no_features_demo = input_demo.shape[1]
    no_output = output.shape[1]
    logging.debug(f"no of lab features= {no_features_lab} \n")
    logging.debug(f"no of vitals features= {no_features_vitals} \n")
    logging.debug(f"no of demographics features= {no_features_demo} \n")
    logging.debug(f"no of output features= {no_output} \n")
    # get the total number of sequences used for the model training
    no_of_sequences = int(input_lab.shape[0] / window_length)
    logging.debug(f"Total number of sequences with length= {window_length} equals to {no_of_sequences} \n")
    # get the list of the output lab values' names
    output_columns_names = output.columns.tolist()
    # reshape the input and output
    input_lab_reshaped = input_lab.values.reshape(no_of_sequences, window_length, no_features_lab)
    input_vitals_reshaped = input_vitals.values.reshape(no_of_sequences, window_length, no_features_vitals)
    input_demo_reshaped = input_demo.values.reshape(no_of_sequences, no_features_demo)
    output_reshaped = output.values.reshape(no_of_sequences, no_output)
    # Split the data
    x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test = train_test_split(
        input_lab_reshaped, input_vitals_reshaped, input_demo_reshaped,
        output_reshaped, test_size=testing_percent, shuffle=True, random_state=42)

    logging.debug(f"Train lab values shape = {x_l_train.shape}")
    logging.debug(f"Test lab values shape = {x_l_test.shape}")
    logging.debug(f"Train lab vitals shape = {x_v_train.shape}")
    logging.debug(f"Test lab vitals shape = {x_v_test.shape}")
    logging.debug(f"Train demographics shape = {x_d_train.shape}")
    logging.debug(f"Test demographics shape = {x_d_test.shape}")
    logging.debug(f"Train output lab values shape = {y_train.shape}")
    logging.debug(f"Test output lab values shape = {y_test.shape}")
    # set the types for the numpy arrays to avoid compatibility issues
    x_l_train = np.asarray(x_l_train).astype('float32')
    x_l_test = np.asarray(x_l_test).astype('float32')
    x_v_train = np.asarray(x_v_train).astype('float32')
    x_v_test = np.asarray(x_v_test).astype('float32')
    x_d_train = np.asarray(x_d_train).astype('float32')
    x_d_test = np.asarray(x_d_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    return x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names


if __name__ == '__main__':
    x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names = get_training_data_mimic(
        window_length=5, testing_percent=0.1)
