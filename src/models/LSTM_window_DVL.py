"""

* Copyright (C) RWTH Aachen University, - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
# This file builds and trains a basic LSTM neural network on the lab values data.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, LSTM, concatenate, BatchNormalization, Dropout, LeakyReLU, ReLU
from tensorflow.keras import backend as K



# CNN+FCN Model
def build(window_length,f1_threshold, no_features_lab=25, num_classes=25, no_features_vitals=12, no_demo_features=3,  nb_cells_1_1=30, nb_cells_2_1=15, nb_cells_1_2=50, nb_cells_2_2=40, nb_cells_1_3=55, nb_cells_2_3=10, dropout_rate_1=0.3, dropout_rate_2=0.5, nb_dense_units_2=30, nb_dense_units_3=120, dropout_rate_final=0.4, learning_rate=0.0011):

    input_layer_l = Input(shape=(window_length, no_features_lab), name="lab_Input")
    #input_layer_v = Input(shape=(window_length, no_features_vitals), name="vitals_Input")
    input_layer_d = Input(shape=(no_demo_features,), name="demographic_Input")

    #input_concat = concatenate([input_layer_l, input_layer_v], axis=1, name='concatenate')

    LSTM_11 = LSTM(nb_cells_1_1, name='LSTM_1_1', return_sequences=True)(input_layer_l)
    activation_11 = LeakyReLU()(LSTM_11)
    batch_norm_11 = BatchNormalization()(activation_11)
    drop_11 = Dropout(dropout_rate_1)(batch_norm_11)
    # pooling_1_1 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_11)
    LSTM_12 = (LSTM(nb_cells_1_2, name='LSTM_1_2', return_sequences=True)(drop_11))
    activation_12 = LeakyReLU()(LSTM_12)
    batch_norm_12 = BatchNormalization()(activation_12)
    drop_12 = Dropout(dropout_rate_1)(batch_norm_12)
    # pooling_1_2 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_12)
    LSTM_13 = (LSTM(nb_cells_1_3, name='LSTM_1_3', return_sequences=False)(drop_12))
    activation_13 = LeakyReLU()(LSTM_13)
    batch_norm_13 = BatchNormalization()(activation_13)
    drop_13 = Dropout(dropout_rate_1)(batch_norm_13)

    flatten_1 = Flatten()(drop_13)


    conc_cnn = concatenate([flatten_1, input_layer_d], axis=1, name='concatenate')

    DENSE_2 = Dense(nb_dense_units_2, name='Dense_2')(conc_cnn)
    activation_31 = LeakyReLU()(DENSE_2)
    batch_norm_y = BatchNormalization()(activation_31)

    DENSE_3 = (Dense(nb_dense_units_3, name='Dense_3'))(batch_norm_y)
    activation_32 = LeakyReLU()(DENSE_3)
    batch_norm_y = BatchNormalization()(activation_32)
    DROP_3 = Dropout(rate=dropout_rate_final)(batch_norm_y)

    output = (Dense(num_classes, activation='sigmoid', name='Output_layer'))(DROP_3)

    model = Model(inputs=[input_layer_l, input_layer_d], outputs=output)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=f1_threshold),
                           tf.keras.metrics.Recall(thresholds=f1_threshold),f1])
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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



if __name__ == "__main__":
    window_length = 5
    no_features_lab = 25
    no_features_vitals = 12
    no_demo_features = 3
    num_classes = 25
    epochs = 300
    f1_threshold = 0.4
    # for window_length in range(4,14):
    model = build(window_length,f1_threshold)
