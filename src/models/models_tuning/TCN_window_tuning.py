
"""

* Copyright (C) Clinomic, GmbH - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
# This file uses the KERAS TUNER library to perform autoML and optimize the architecture of the Simple LSTM model.

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Masking,Input,concatenate,Dropout
from tensorflow.keras.models import Model
import keras_tuner as kt
from keras_tuner.tuners import BayesianOptimization
import pickle
import numpy as np
from tensorflow.keras import backend as K
from tcn import TCN


cwd=os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(cwd, "../saved_models", "tuner_TCN.h5")
log_dir_arch =os.path.join(cwd, "../..", '..', "data", "logs", "tuner", "TCN")

gpus = tf.config.experimental.list_physical_devices('GPU')
from get_training_data import get_training_data_mimic




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

    nb_filters_1 = hp.Int(f"TCN_nb_filters_1", min_value=10, max_value=200, step=10)
    nb_filters_2 = hp.Int(f"TCN_nb_filters_2", min_value=10, max_value=200, step=10)
    nb_dense_units=hp.Int("Desnse layer units",min_value=10, max_value=200, step=10)
    nb_stacks_1 = hp.Choice('nb_stacks_1', values=[1, 2])
    nb_stacks_2 = hp.Choice('nb_stacks_2', values=[1, 2])
    use_skip_connections = hp.Choice('S.C.1', values=[True, False])
    dropout_rate_1 = hp.Choice('drop out rate 1', values=[0.2, 0.3, 0.4])
    dropout_rate_2 = hp.Choice('drop out rate 2', values=[0.2, 0.3, 0.4])
    dropout_rate_final = hp.Choice('Final drop out rate', values=[0.2, 0.3, 0.4, 0.5])
    use_batch_norm = hp.Choice('B.N', values=[True, False])



    input_layer_1 = Input(shape=(window_length, no_features), name="Lab_Input")
    input_layer_2 = Input(shape=(no_demo_features,), name="DEMOGRAPHICS_Input")


    DENSE_1 = Dense(nb_dense_units, activation='relu', name='Dense_1')(input_layer_2)
    model_1 = Model(inputs=input_layer_2, outputs=DENSE_1)


    masking_layer=Masking(mask_value=0.0)(input_layer_1)
    convolved_1 = TCN(nb_filters=nb_filters_1, nb_stacks=nb_stacks_1, padding='causal',
                      use_skip_connections=use_skip_connections, dropout_rate=dropout_rate_1, return_sequences=True,
                      use_batch_norm=use_batch_norm)(masking_layer)
    convolved_2 = TCN(nb_filters=nb_filters_2, nb_stacks=nb_stacks_2, padding='causal',
                      use_skip_connections=use_skip_connections, dropout_rate=dropout_rate_2, return_sequences=False,
                      use_batch_norm=use_batch_norm)(convolved_1)

    model_2 = Model(inputs=input_layer_1, outputs=convolved_2)
    conc = concatenate([model_2.output, model_1.output], axis=1, name='concatenate')


    DENSE_2 = Dense(hp.Int(f"dense_2_layer", min_value=50, max_value=200, step=10), activation='relu', name='Dense_2')(conc)
    DENSE_3 = (Dense(hp.Int(f"dense_3_layer", min_value=50, max_value=200, step=10), activation='relu', name='Dense_3'))(DENSE_2)
    DROP = Dropout(rate=dropout_rate_final)(DENSE_3)

    output = (Dense(num_classes, activation='sigmoid', name='Output_layer'))(DROP)

    model = Model(inputs=[input_layer_1, input_layer_2], outputs=output)
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
        project_name="TCN_Window_TRIAL_w5_L1",
        executions_per_trial=2
    )

    print(tuner.search_space_summary())


    tuner.search(x=[x_l_train, x_d_train],y=y_train,
                 epochs=4,
                 validation_split=0.15,
                 verbose=2)


    with open("TCN_Window_DVL_SEARCH_w5_l1.pkl", "wb") as f:
        pickle.dump(tuner, f)

    tuner = pickle.load(open("TCN_Window_DVL_SEARCH_l1.pkl", "rb"))

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
    window_length=6
    no_features = 25
    no_demo_features = 3
    num_classes = 25
    # ================================================
    model=tune_model(window_length)