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
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, concatenate, BatchNormalization, Dropout, LeakyReLU, ReLU
from tf_gpu import tf_gpu
from keras import backend as K
# Define call backs and special functions;
tf_gpu()


# CNN+FCN Model
def build(window_length,f1_threshold, no_features_lab=11,num_classes=11, no_features_vitals=12, no_demo_features=3,
                CNN_kernal_size_11=2,
                CNN_kernal_size_12=3, CNN_kernal_size_21=4, CNN_kernal_size_22=3, CNN_nb_filters_1_1=100,
                CNN_nb_filters_1_2=100, CNN_nb_filters_1_3=100, CNN_nb_filters_2_1=100,
                CNN_nb_filters_2_2=65, CNN_nb_filters_2_3=45, dense_2_n=200, dense_3_n=200, drop_1=0.2,
                drop_2=0.4,
                drop_final=0.2,
                learnin_rate=0.000333):

    input_layer_l = Input(shape=(window_length, no_features_lab), name="lab_Input")
    input_layer_d = Input(shape=(no_demo_features,), name="demographic_Input")

    convolved_11 = (Conv1D(CNN_nb_filters_1_1, CNN_kernal_size_11, strides=1, padding='same', name='CONV_1_1'))(
        input_layer_l)
    activation_11 = LeakyReLU()(convolved_11)
    batch_norm_11 = BatchNormalization()(activation_11)

    # pooling_1_1 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_11)
    convolved_12 = (Conv1D(CNN_nb_filters_1_2, CNN_kernal_size_11, padding="same", strides=1, name='CONV_1_2')(batch_norm_11))
    activation_12 = LeakyReLU()(convolved_12)
    batch_norm_12 = BatchNormalization()(activation_12)

    # pooling_1_2 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_12)
    convolved_13 = (Conv1D(CNN_nb_filters_1_3, CNN_kernal_size_12, padding="same", strides=1, name='CONV_1_3')(batch_norm_12))
    activation_13 = LeakyReLU()(convolved_13)
    batch_norm_13 = BatchNormalization()(activation_13)

    # pooling_1_3 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_13)
    # =================================================================================================================
    convolved_21 = (Conv1D(CNN_nb_filters_2_1, CNN_kernal_size_21, strides=1, padding='same', name='CONV_2_1'))(
        input_layer_l)
    activation_21 = LeakyReLU()(convolved_21)
    batch_norm_21 = BatchNormalization()(activation_21)

    # pooling_2_1 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_21)
    convolved_22 = (Conv1D(CNN_nb_filters_2_2, CNN_kernal_size_21, padding="same", strides=1, name='CONV_2_2')(batch_norm_21))
    activation_22 = LeakyReLU()(convolved_22)
    batch_norm_22 = BatchNormalization()(activation_22)

    # pooling_2_2 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_22)
    convolved_23 = (Conv1D(CNN_nb_filters_2_3, CNN_kernal_size_22, padding="same", strides=1, name='CONV_2_3')(batch_norm_22))
    activation_23 = LeakyReLU()(convolved_23)
    batch_norm_23 = BatchNormalization()(activation_23)

    # pooling_2_3 = MaxPooling1D(pool_size=2, padding="same")(batch_norm_23)

    # DROP_1 = Dropout(rate=0.2)(pooling_1_3)
    # DROP_2 = Dropout(rate=0.2)(pooling_2_3)
    flatten_1 = Flatten()(batch_norm_13)
    flatten_2 = Flatten()(batch_norm_23)

    model_2 = Model(inputs=input_layer_l, outputs=flatten_1)
    model_3 = Model(inputs=input_layer_l, outputs=flatten_2)
    conc_cnn = concatenate([model_2.output, model_3.output, input_layer_d], axis=1, name='concatenate_cnn')
    DENSE_2 = Dense(dense_2_n, name='Dense_2')(conc_cnn)
    activation_31 = ReLU()(DENSE_2)
    batch_norm_y = BatchNormalization()(activation_31)
    DENSE_3 = (Dense(dense_3_n, name='Dense_3'))(batch_norm_y)
    activation_32 = ReLU()(DENSE_3)
    batch_norm_y = BatchNormalization()(activation_32)


    output = (Dense(num_classes, activation='sigmoid', name='Output_layer'))(batch_norm_y)

    model = Model(inputs=[input_layer_l,input_layer_d], outputs=output)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    opt = keras.optimizers.Adam(lr=learnin_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', tf.keras.metrics.Precision(thresholds=f1_threshold),
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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# PrintInfo()


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
