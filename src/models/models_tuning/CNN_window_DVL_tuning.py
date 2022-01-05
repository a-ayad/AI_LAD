
"""

* Copyright (C) Clinomic, GmbH - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
# This file uses the KERAS TUNER library to perform autoML and optimize the architecture of the Simple LSTM model.

import datetime as datetime
import os
import pandas as pd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,LeakyReLU,Input,concatenate,Dropout,Flatten,BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import datetime
from datetime import time

import keras_tuner as kt
from keras_tuner.tuners import RandomSearch, Hyperband,BayesianOptimization
from keras_tuner.engine.hyperparameters import HyperParameters
import pickle
import numpy as np
from tcn import TCN, tcn_full_summary
import sys, pathlib
from tensorflow.keras import backend as K
from get_training_data import get_training_data_mimic, get_training_data_eicu




base_path = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(base_path)
tf.keras.backend.set_floatx('float32')

model_path = os.path.join(base_path,"src", "saved_models", "CNN_Window_TUNER.h5")
log_dir_arch =os.path.join(base_path, "data","logs","tuner","CNN_Window_DVL")

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
    nb_filters_1_1 = hp.Int(f"CNN_nb_filters_1_1", min_value=5, max_value=100, step=5)
    nb_filters_1_2 = hp.Int(f"CNN_nb_filters_1_2", min_value=5, max_value=100, step=5)
    nb_filters_1_3 = hp.Int(f"CNN_nb_filters_1_3", min_value=5, max_value=100, step=5)
    nb_filters_2_1 = hp.Int(f"CNN_nb_filters_2_1", min_value=5, max_value=100, step=5)
    nb_filters_2_2 = hp.Int(f"CNN_nb_filters_2_2", min_value=5, max_value=100, step=5)
    nb_filters_2_3 = hp.Int(f"CNN_nb_filters_2_3", min_value=5, max_value=100, step=5)
    nb_filters_3_1 = hp.Int(f"CNN_nb_filters_3_1", min_value=5, max_value=100, step=5)
    nb_filters_3_2 = hp.Int(f"CNN_nb_filters_3_2", min_value=5, max_value=100, step=5)
    nb_filters_3_3 = hp.Int(f"CNN_nb_filters_3_3", min_value=5, max_value=100, step=5)

    kernel_size_1_1=hp.Int(f"CNN_kernal_size_11", min_value=1, max_value=5, step=1)
    kernel_size_1_2=hp.Int(f"CNN_kernal_size_12", min_value=1, max_value=5, step=1)
    kernel_size_1_3 = hp.Int(f"CNN_kernal_size_13", min_value=1, max_value=5, step=1)
    kernel_size_2_1=hp.Int(f"CNN_kernal_size_21", min_value=1, max_value=5, step=1)
    kernel_size_2_2=hp.Int(f"CNN_kernal_size_22", min_value=1, max_value=5, step=1)
    kernel_size_2_3 = hp.Int(f"CNN_kernal_size_23", min_value=1, max_value=5, step=1)
    kernel_size_3_1 = hp.Int(f"CNN_kernal_size_31", min_value=1, max_value=5, step=1)
    kernel_size_3_2 = hp.Int(f"CNN_kernal_size_32", min_value=1, max_value=5, step=1)
    kernel_size_3_3 = hp.Int(f"CNN_kernal_size_33", min_value=1, max_value=5, step=1)



    nb_dense_units_2 = hp.Int("2nd Dense layer units", min_value=10, max_value=200, step=10)
    nb_dense_units_3 = hp.Int("3rd Dense layer units", min_value=10, max_value=200, step=10)
    nb_dense_units_4 = hp.Int("4th Dense layer units", min_value=10, max_value=200, step=10)

    dropout_rate_final = hp.Choice('Final drop out rate', values=[0.2, 0.3, 0.4, 0.5])
    input_layer_l = Input(shape=(window_length, no_features_lab), name="lab_Input")
    #input_layer_v = Input(shape=(window_length, no_features_vitals), name="vitals_Input")
    input_layer_d = Input(shape=(no_demo_features,), name="demographic_Input")

    # model_1 = Model(inputs=input_layer_2, outputs=DENSE_1)

    # masking_layer=Masking(mask_value=0.0)(input_layer_1)

    convolved_11 = (Conv1D(nb_filters_1_1, kernel_size_1_1, strides=1, padding='same', name='CONV_1_1'))(input_layer_l)
    activation_11 = LeakyReLU()(convolved_11)
    batch_norm_11 = BatchNormalization()(activation_11)



    convolved_12 = (Conv1D(nb_filters_1_2, kernel_size_1_2, padding="same", strides=1, name='CONV_1_2')(batch_norm_11))
    activation_12 = LeakyReLU()(convolved_12)
    batch_norm_12 = BatchNormalization()(activation_12)



    convolved_13 = (Conv1D(nb_filters_1_3, kernel_size_1_3, padding="same", strides=1, name='CONV_1_3')(batch_norm_12))
    activation_13 = LeakyReLU()(convolved_13)
    batch_norm_13 = BatchNormalization()(activation_13)


    # pooling_1_3 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_13)
    # =================================================================================================================
    convolved_21 = (Conv1D(nb_filters_2_1, kernel_size_2_1, strides=1, padding='same', name='CONV_2_1'))(input_layer_l)
    activation_21 = LeakyReLU()(convolved_21)
    batch_norm_21 = BatchNormalization()(activation_21)



    convolved_22 = (Conv1D(nb_filters_2_2, kernel_size_2_2, padding="same", strides=1, name='CONV_2_2')(batch_norm_21))
    activation_22 = LeakyReLU()(convolved_22)
    batch_norm_22 = BatchNormalization()(activation_22)



    convolved_23 = (Conv1D(nb_filters_2_3, kernel_size_2_3, padding="same", strides=1, name='CONV_2_3')(batch_norm_22))
    activation_23 = LeakyReLU()(convolved_23)
    batch_norm_23 = BatchNormalization()(activation_23)

    # pooling_1_3 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_13)
    # =================================================================================================================
    convolved_31 = (Conv1D(nb_filters_3_1, kernel_size_3_1, strides=1, padding='same', name='CONV_3_1'))(input_layer_l)
    activation_31 = LeakyReLU()(convolved_31)
    batch_norm_31 = BatchNormalization()(activation_31)

    convolved_32 = (Conv1D(nb_filters_3_2, kernel_size_3_2, padding="same", strides=1, name='CONV_3_2')(batch_norm_31))
    activation_32 = LeakyReLU()(convolved_32)
    batch_norm_32 = BatchNormalization()(activation_32)

    convolved_33 = (Conv1D(nb_filters_3_3, kernel_size_3_3, padding="same", strides=1, name='CONV_3_3')(batch_norm_32))
    activation_33 = LeakyReLU()(convolved_33)
    batch_norm_33 = BatchNormalization()(activation_33)


    # DROP_1 = Dropout(rate=0.2)(pooling_1_3)
    # DROP_2 = Dropout(rate=0.2)(pooling_2_3)

    flatten_1 = Flatten()(batch_norm_13)
    flatten_2 = Flatten()(batch_norm_23)
    flatten_3 = Flatten()(batch_norm_33)

    model_1 = Model(inputs=input_layer_l, outputs=flatten_1)
    model_2 = Model(inputs=input_layer_l, outputs=flatten_2)
    model_3 = Model(inputs=input_layer_l, outputs=flatten_3)

    conc_cnn = concatenate([model_1.output, model_2.output, model_3.output, input_layer_d], axis=1, name='concatenate_cnn')

    DENSE_2 = Dense(nb_dense_units_2, name='Dense_2')(conc_cnn)
    activation_31 = LeakyReLU()(DENSE_2)
    batch_norm_y = BatchNormalization()(activation_31)

    DENSE_3 = (Dense(nb_dense_units_3, name='Dense_3'))(batch_norm_y)
    activation_32 = LeakyReLU()(DENSE_3)
    batch_norm_y = BatchNormalization()(activation_32)

    DENSE_4 = Dense(nb_dense_units_4, name='Dense_4')(batch_norm_y)
    activation_41 = LeakyReLU()(DENSE_4)
    batch_norm_y = BatchNormalization()(activation_41)

    DROP_3 = Dropout(rate=dropout_rate_final)(batch_norm_y)
    output = (Dense(num_classes, activation='sigmoid', name='Output_layer'))(DROP_3)

    model = Model(inputs=[input_layer_l, input_layer_d], outputs=output)
    #==============================================================================================

    opt = keras.optimizers.Adam(hp.Float(
        'learning_rate',
        min_value=1e-5,
        max_value=1e-2,
        sampling='LOG',
        default=1e-3
    ))
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=0.4),
                           tf.keras.metrics.Recall(thresholds=0.4),f1])

    return model






def tune_model(window_length):
    x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names = get_training_data_mimic(
        window_length=6, testing_percent=0.1)
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
        objective=kt.Objective("val_f1", direction="max"),
        max_trials=100,
        seed=42,
        directory=log_dir_arch,
        project_name="CNN_Window_TRIAL_w6_L3",
        executions_per_trial=2
    )

    print(tuner.search_space_summary())


    tuner.search(x=[x_l_train, x_d_train],y=y_train,
                 epochs=4,
                 validation_split=0.15,
                 verbose=2)


    with open("CNN_Window_DVL_SEARCH_l2.pkl", "wb") as f:
        pickle.dump(tuner, f)

    tuner = pickle.load(open("CNN_Window_DVL_SEARCH_l2.pkl", "rb"))

    print(tuner.get_best_hyperparameters()[0].values)
    model = tuner.get_best_models()
    print(tuner.results_summary())
    print(tuner.get_best_models()[0].summary())
    return model



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




if __name__ == "__main__":
    window_length = 5
    no_features_lab=25
    no_features_vitals=12
    no_demo_features = 3
    num_classes = 25
    # ================================================
    model=tune_model(window_length)