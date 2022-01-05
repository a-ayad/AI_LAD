"""

* Copyright (C) Clinomic, GmbH - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
# This file builds and trains a basic LSTM neural network on the lab values data.
# comment : The training should be changed to using tensorflow gradient tape

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LeakyReLU, Flatten, Dense, GRU, BatchNormalization, concatenate,\
    Bidirectional, Masking, Lambda, Dropout
import os
from tensorflow.keras import backend as K
import datetime
from TransformerModel import TSTransformer
from tensorflow_addons.optimizers import RectifiedAdam

cwd = os.path.dirname(os.path.abspath(__file__))
log_date = datetime.datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.join(cwd, "..", '..', "data", "logs", "fit", log_date)
tf.keras.backend.set_floatx('float32')


# CNN+FCN Model
def build(window_length, f1_threshold, learning_rate=1e-3, no_features_lab=25, num_classes=25, no_features_vitals=12,
          no_demo_features=3,
          time2vec_dim=5, num_heads=10, head_size=32, ff_dim=4, num_layers=5, dropout=0.3, nb_dense_units_2=200,
          nb_dense_units_3=200, dropout_rate_final=0.3):

    input_layer_l = Input(shape=(window_length, no_features_lab), name="lab_Input")
    # input_layer_v = Input(shape=(window_length, no_features_vitals), name="vitals_Input")
    input_layer_d = Input(shape=(no_demo_features,), name="demographic_Input")

    ts = TSTransformer(time2vec_dim=time2vec_dim, num_heads=num_heads, head_size=head_size, ff_dim=ff_dim,
                       num_layers=num_layers, dropout=dropout)(input_layer_l)

    # pooling_2_3 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_23)

    # DROP_1 = Dropout(rate=0.2)(pooling_1_3)
    # DROP_2 = Dropout(rate=0.2)(pooling_2_3)

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

    opt = RectifiedAdam(learning_rate=learning_rate)
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
    WINDOW_LENGTH = 5
    F1_THRESHOLD = 0.4
    LEARNING_RATE = 0.0011
    model = build(WINDOW_LENGTH, F1_THRESHOLD)
