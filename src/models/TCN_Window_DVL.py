"""

* Copyright (C) RWTH Aachen University, - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
# This file builds and trains a basic LSTM neural network on the lab values data.
# comment : The training should be changed to using tensorflow gradient tape

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LeakyReLU, Flatten, Dense, GRU, BatchNormalization, concatenate, Dropout
from tensorflow.keras import backend as K
from tcn import TCN, tcn_full_summary


# CNN+FCN Model
def build(window_length, f1_threshold, learning_rate=0.01, no_features_lab=25, num_classes=25, no_features_vitals=12,
          no_demo_features=3,
          nb_cells_1_1=200,
          nb_cells_1_2=60,
          dropout_rate_1=0.3,
          nb_dense_units_2=160,
          nb_dense_units_3=200,
          dropout_rate_final=0.2):
    input_layer_l = Input(shape=(window_length, no_features_lab), name="lab_Input")
    #input_layer_v = Input(shape=(window_length, no_features_vitals), name="vitals_Input")
    input_layer_d = Input(shape=(no_demo_features,), name="demographic_Input")

    # model_1 = Model(inputs=input_layer_2, outputs=DENSE_1)

    # masking_layer=Masking(mask_value=0.0)(input_layer_1)

    TCN_11 = TCN(nb_filters=nb_cells_1_1, nb_stacks=2, padding='causal',
                 use_skip_connections=True, return_sequences=True,
                 use_batch_norm=False)(input_layer_l)
    activation_11 = LeakyReLU()(TCN_11)
    batch_norm_11 = BatchNormalization()(activation_11)
    drop_11 = Dropout(dropout_rate_1)(batch_norm_11)

    TCN_12 = TCN(nb_filters=nb_cells_1_2, nb_stacks=2, padding='causal',
                 use_skip_connections=True, return_sequences=False,
                 use_batch_norm=False)(drop_11)
    activation_12 = LeakyReLU()(TCN_12)
    batch_norm_12 = BatchNormalization()(activation_12)
    drop_12 = Dropout(dropout_rate_1)(batch_norm_12)




    flatten_1 = Flatten()(drop_12)

    model_2 = Model(inputs=input_layer_l, outputs=flatten_1)

    conc_cnn = concatenate([model_2.output, input_layer_d], axis=1, name='concatenate')

    DENSE_2 = Dense(nb_dense_units_2, name='Dense_2')(conc_cnn)
    activation_31 = LeakyReLU()(DENSE_2)
    batch_norm_y = BatchNormalization()(activation_31)

    DENSE_3 = (Dense(nb_dense_units_3, name='Dense_3'))(batch_norm_y)
    activation_32 = LeakyReLU()(DENSE_3)
    batch_norm_y = BatchNormalization()(activation_32)

    DROP_3 = Dropout(rate=dropout_rate_final)(batch_norm_y)
    output = (Dense(num_classes, activation='sigmoid', name='Output_layer'))(DROP_3)

    model = Model(inputs=[input_layer_l, input_layer_d], outputs=output)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1,
                                                                decay_steps=100,
                                                                decay_rate=0.9)

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=f1_threshold),
                           tf.keras.metrics.Recall(thresholds=f1_threshold), f1])
    model.summary()
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
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == "__main__":
    window_length = 5
    f1_threshold = 0.4

    model = build(window_length, f1_threshold)
