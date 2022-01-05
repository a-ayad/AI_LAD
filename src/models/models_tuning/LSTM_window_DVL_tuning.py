
"""

* Copyright (C) Clinomic, GmbH - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
# This file uses the KERAS TUNER library to perform autoML and optimize the architecture of the Simple LSTM model.

import datetime as datetime
from logzero import logger
import os
import pandas as pd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras.layers import Dense,LSTM,GRU,Bidirectional,LeakyReLU,Input,concatenate,Dropout,Flatten,BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import datetime
from datetime import time
from kerastuner.tuners import RandomSearch, Hyperband,BayesianOptimization
import kerastuner
from kerastuner.engine.hyperparameters import HyperParameters
import pickle
import numpy as np
from tcn import TCN, tcn_full_summary



cwd=os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(cwd, "../saved_models", "LSTM_Window_TUNER.h5")
log_dir_arch =os.path.join(cwd, "../..", '..', "data", "logs", "tuner", "LSTM_Window_DVL")

gpus = tf.config.experimental.list_physical_devices('GPU')


if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)






def build_model(hp):
    nb_cells_1_1 = hp.Int(f"LSTM_cells_1_1", min_value=5, max_value=100, step=5)
    nb_cells_2_1 = hp.Int(f"LSTM_cells_2_1", min_value=5, max_value=100, step=5)
    nb_cells_1_2 = hp.Int(f"LSTM_cells_1_2", min_value=5, max_value=100, step=5)
    nb_cells_2_2 = hp.Int(f"LSTM_cells_2_2", min_value=5, max_value=100, step=5)
    nb_cells_1_3 = hp.Int(f"LSTM_cells_1_3", min_value=5, max_value=100, step=5)
    nb_cells_2_3 = hp.Int(f"LSTM_cells_2_3", min_value=5, max_value=100, step=5)


    dropout_rate_1 = hp.Choice('drop out rate 1', values=[0.2, 0.3, 0.4])
    dropout_rate_2 = hp.Choice('drop out rate 2', values=[0.2, 0.3, 0.4])

    nb_dense_units_2 = hp.Int("2nd Dense layer units", min_value=10, max_value=200, step=10)
    nb_dense_units_3 = hp.Int("3rd Dense layer units", min_value=10, max_value=200, step=10)

    dropout_rate_final = hp.Choice('Final drop out rate', values=[0.2, 0.3, 0.4, 0.5])

    input_layer_l = Input(shape=(window_length, no_features_lab), name="lab_Input")
    input_layer_v = Input(shape=(window_length, no_features_vitals), name="vitals_Input")
    input_layer_d = Input(shape=(no_demo_features,), name="demographic_Input")

    # model_1 = Model(inputs=input_layer_2, outputs=DENSE_1)

    # masking_layer=Masking(mask_value=0.0)(input_layer_1)

    LSTM_11 = (LSTM(nb_cells_1_1, return_sequences=True))(input_layer_l)
    activation_11 = LeakyReLU()(LSTM_11)
    batch_norm_11 = BatchNormalization()(activation_11)
    drop_11 = Dropout(dropout_rate_1)(batch_norm_11)
    # pooling_1_1 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_11)
    LSTM_12 = (LSTM(nb_cells_1_2, return_sequences=True)(drop_11))
    activation_12 = LeakyReLU()(LSTM_12)
    batch_norm_12 = BatchNormalization()(activation_12)
    drop_12 = Dropout(dropout_rate_1)(batch_norm_12)
    # pooling_1_2 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_12)
    LSTM_13 = (LSTM(nb_cells_1_3, return_sequences=False)(drop_12))
    activation_13 = LeakyReLU()(LSTM_13)
    batch_norm_13 = BatchNormalization()(activation_13)
    drop_13 = Dropout(dropout_rate_1)(batch_norm_13)
    # pooling_1_3 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_13)
    # =================================================================================================================
    LSTM_21 = (LSTM(nb_cells_2_1, return_sequences=True))(input_layer_v)
    activation_21 = LeakyReLU()(LSTM_21)
    batch_norm_21 = BatchNormalization()(activation_21)
    drop_21 = Dropout(dropout_rate_2)(batch_norm_21)
    # pooling_2_1 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_21)
    LSTM_22 = (LSTM(nb_cells_2_2, return_sequences=True)(drop_21))
    activation_22 = LeakyReLU()(LSTM_22)
    batch_norm_22 = BatchNormalization()(activation_22)
    drop_22 = Dropout(dropout_rate_2)(batch_norm_22)
    # pooling_2_2 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_22)
    LSTM_23 = (LSTM(nb_cells_2_3, return_sequences=False)(drop_22))
    activation_23 = LeakyReLU()(LSTM_23)
    batch_norm_23 = BatchNormalization()(activation_23)
    drop_23 = Dropout(dropout_rate_2)(batch_norm_23)
    # pooling_2_3 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_23)

    # DROP_1 = Dropout(rate=0.2)(pooling_1_3)
    # DROP_2 = Dropout(rate=0.2)(pooling_2_3)
    flatten_1 = Flatten()(drop_13)
    flatten_2 = Flatten()(drop_23)

    model_2 = Model(inputs=input_layer_l, outputs=flatten_1)
    model_3 = Model(inputs=input_layer_v, outputs=flatten_2)
    conc_cnn = concatenate([model_2.output, model_3.output, input_layer_d], axis=1, name='concatenate_cnn')
    DENSE_2 = Dense(nb_dense_units_2, name='Dense_2')(conc_cnn)
    activation_31 = LeakyReLU()(DENSE_2)
    batch_norm_y = BatchNormalization()(activation_31)

    DENSE_3 = (Dense(nb_dense_units_3, name='Dense_3'))(batch_norm_y)
    activation_32 = LeakyReLU()(DENSE_3)
    batch_norm_y = BatchNormalization()(activation_32)

    DROP_3 = Dropout(rate=dropout_rate_final)(batch_norm_y)
    output = (Dense(num_classes, activation='sigmoid', name='Output_layer'))(DROP_3)

    model = Model(inputs=[input_layer_l, input_layer_v, input_layer_d], outputs=output)
    #==============================================================================================

    opt = keras.optimizers.Adam(hp.Float(
        'learning_rate',
        min_value=1e-5,
        max_value=1e-2,
        sampling='LOG',
        default=1e-3
    ))
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', tf.keras.metrics.Precision(thresholds=0.4),
                           tf.keras.metrics.Recall(thresholds=0.4)])

    return model

def tune_model(window_length):
    train_x_l = np.load(
        os.path.join(cwd, "../..", '..', "data", "lab_values", "windowed_data_DVL", f"train_x_l_WL{window_length}.npy"),
        allow_pickle=True)
    train_x_v = np.load(
        os.path.join(cwd, "../..", '..', "data", "lab_values", "windowed_data_DVL", f"train_x_v_WL{window_length}.npy"),
        allow_pickle=True)
    train_x_d = np.load(
        os.path.join(cwd, "../..", '..', "data", "lab_values", "windowed_data_DVL", f"train_x_d_WL{window_length}.npy"),
        allow_pickle=True)
    train_y = np.load(
        os.path.join(cwd, "../..", '..', "data", "lab_values", "windowed_data_DVL", f"train_y_WL{window_length}.npy"),
        allow_pickle=True)
    train_x_l = np.asarray(train_x_l).astype('float32')
    train_x_v = np.asarray(train_x_v).astype('float32')
    train_x_d = np.asarray(train_x_d).astype('float32')
    train_y = np.asarray(train_y).astype('float32')
    print(train_x_l.shape)
    print(train_x_v.shape)
    print(train_x_d.shape)
    print(train_y.shape)
    # ================================================
    '''
    tuner = RandomSearch(
        build_model,
        objective=kerastuner.Objective("val_accuracy", direction="max"),
        max_trials=2,
        directory=log_dir_arch,
        project_name="BASIC_LSTM_TRIAL_2"
    )
    '''

    tuner= BayesianOptimization(
        build_model,
        objective=kerastuner.Objective("val_accuracy", direction="max"),
        max_trials=100,
        seed=42,
        directory=log_dir_arch,
        project_name="LSTM_Window_TRIAL",
        executions_per_trial=1
    )

    print(tuner.search_space_summary())


    tuner.search(x=[train_x_l, train_x_v, train_x_d],y=train_y,
                 epochs=4,
                 validation_split=0.15,
                 verbose=2)


    with open("LSTM_Window_DVL_SEARCH.pkl", "wb") as f:
        pickle.dump(tuner, f)

    tuner = pickle.load(open("LSTM_Window_DVL_SEARCH.pkl", "rb"))

    print(tuner.get_best_hyperparameters()[0].values)
    model = tuner.get_best_models()
    print(tuner.results_summary())
    print(tuner.get_best_models()[0].summary())
    return model



if __name__ == "__main__":
    window_length = 5
    no_features_lab=25
    no_features = 37
    no_features_vitals=no_features-no_features_lab
    no_demo_features = 3
    num_classes = 25
    # ================================================
    model=tune_model(window_length)