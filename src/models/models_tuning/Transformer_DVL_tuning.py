
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
from tensorflow.keras.layers import Dense,Conv1D, LeakyReLU,Input,concatenate,Dropout,Flatten,BatchNormalization
from tensorflow.keras.models import Model
import keras_tuner as kt
from keras_tuner.tuners import BayesianOptimization
import pickle
import sys, pathlib
from tensorflow.keras import backend as K
from get_training_data import get_training_data_mimic
from TransformerModel import TSTransformer
from tensorflow_addons.optimizers import RectifiedAdam




base_path = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(base_path)
tf.keras.backend.set_floatx('float32')

model_path = os.path.join(base_path,"src", "saved_models", "Transformer_Window_TUNER.h5")
log_dir_arch =os.path.join(base_path, "data","logs","tuner","Transformer_Window_DVL")

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
    #time2vec_dim = hp.Int(f"time2vec_dim", min_value=1, max_value=5, step=1)
    num_heads = hp.Int(f"num_heads", min_value=1, max_value=10, step=1)
    head_size = hp.Choice(f"head_size", values=[32, 64, 128, 256])
    #ff_dim = hp.Choice(f"ff_dim", values=[0, 1, 2, 4])
    num_layers = hp.Int(f"num_layers", min_value=1, max_value=5, step=1)
    dropout = hp.Choice('attention drop out rate', values=[0.2, 0.3, 0.4, 0.5])
    nb_dense_units_2 = hp.Int("2nd Dense layer units", min_value=10, max_value=200, step=10)
    nb_dense_units_3 = hp.Int("3rd Dense layer units", min_value=10, max_value=200, step=10)
    dropout_rate_final = hp.Choice('Final drop out rate', values=[0.2, 0.3, 0.4, 0.5])






    input_layer_l = Input(shape=(window_length, no_features_lab), name="lab_Input")
    #input_layer_v = Input(shape=(window_length, no_features_vitals), name="vitals_Input")
    input_layer_d = Input(shape=(no_demo_features,), name="demographic_Input")

    ts = TSTransformer(time2vec_dim=1, num_heads=num_heads, head_size=head_size, ff_dim=None,
                       num_layers=num_layers,
                       dropout=dropout)(input_layer_l)
    conc_cnn = concatenate([ts, input_layer_d], axis=1, name='concatenate_output')

    DENSE_2 = Dense(nb_dense_units_2, name='Dense_2')(conc_cnn)
    activation_31 = LeakyReLU()(DENSE_2)
    batch_norm_y = BatchNormalization()(activation_31)
    DENSE_3 = (Dense(nb_dense_units_3, name='Dense_3'))(batch_norm_y)
    activation_32 = LeakyReLU()(DENSE_3)
    batch_norm_y = BatchNormalization()(activation_32)
    DROP_3 = Dropout(rate=dropout_rate_final)(batch_norm_y)
    output = (Dense(num_classes, activation='sigmoid', name='Output_layer'))(DROP_3)

    model = Model(inputs=[input_layer_l, input_layer_d], outputs=output)
    #==============================================================================================

    opt = RectifiedAdam(hp.Float(
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
        project_name="Transformer_Window_TRIAL",
        executions_per_trial=2
    )

    print(tuner.search_space_summary())


    tuner.search(x=[x_l_train, x_d_train],y=y_train,
                 epochs=4,
                 validation_split=0.15,
                 verbose=2)


    with open("Transformer_Window_DVL_SEARCH.pkl", "wb") as f:
        pickle.dump(tuner, f)

    tuner = pickle.load(open("Transformer_Window_DVL_SEARCH.pkl", "rb"))

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
    window_length = 6
    no_features_lab=25
    no_features_vitals=12
    no_demo_features = 3
    num_classes = 25
    # ================================================
    model=tune_model(window_length)