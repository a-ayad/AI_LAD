"""

* Copyright (C) RWTH Aachen University, - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""

# Assures that imports are found regardless of the base_path (Terminal) or IDE used
import pathlib
import os
import random
import pandas as pd

# Framework imports
from src.models import CNN_Window_DVL, Transformer_Window_DVL, LSTM_window_DVL, TCN_Window_DVL
from get_training_data import get_training_data_mimic, get_training_data_eicu

# other imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import sklearn.metrics as skm

# imports for weights and biasis
import wandb
from wandb.keras import WandbCallback

# Set the paths
base_path = pathlib.Path(__file__).resolve().parents[1]
# sys.path.append(base_path)

# set the backend float type for keras
tf.keras.backend.set_floatx('float32')

# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2 ** 32 - 1)

def tf_gpu():

# to solve the gpu assigment issue
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)






def generate_training_data_mimic(window_length, testing_percent):
    x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names = get_training_data_mimic(
        window_length,
        testing_percent)
    return x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names


def generate_training_data_eicu(window_length, testing_percent):
    x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names = get_training_data_eicu(
        window_length=window_length, testing_percent=testing_percent)
    return x_l_train, x_l_test, x_v_train, x_v_test, x_d_train, x_d_test, y_train, y_test, output_columns_names


def train_model(model, EPOCHS, VALIDATION_SPLIT, BATCH_SIZE, x_l_train, x_v_train, x_d_train, y_train):
    print("training the model")

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, min_delta=0.01)
    history = model.fit(x=[x_l_train, x_d_train], y=y_train, batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT, epochs=EPOCHS,
                        callbacks=[WandbCallback(), early_stopping], verbose=1)

    return model, history


def save_model(model, model_name, history, window_length, results_1, results_2):
    # Save model's weight
    model_name += "_WL{}".format(window_length)
    history_file_name = model_name + '.csv'
    saved_model_name = model_name
    mimic_file_name = model_name + '_mimic.csv'
    eicu_file_name = model_name + '_eicu.csv'
    history_file_path = os.path.join(base_path, "data", "model_history", history_file_name)
    mimic_testing_file_path = os.path.join(base_path, "data", "models_testing_results", mimic_file_name)
    eicu_testing_file_path = os.path.join(base_path, "data", "models_testing_results", eicu_file_name)
    model_file_path = os.path.join(base_path, "src", "models", "saved_models", saved_model_name)
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)
    with open(history_file_path, mode='w') as f:
        hist_df.to_csv(f)
        print("History saved successfully")
    with open(mimic_testing_file_path, mode='w') as f:
        results_1.to_csv(f)
        print("mimic testing results saved successfully")
    with open(eicu_testing_file_path, mode='w') as f:
        results_2.to_csv(f)
        print("eicu testing saved successfully")
    try:

        model.save_weights(model_file_path)
        print("model saved successfully")
    except:
        raise ValueError("Model cannot be saved")

    # save training history to csv:


def test_on_mimic(model, window_length, f1_threshold, x_l_test, x_v_test, x_d_test, y_test, output_columns_names):
    # test the model on the test chunk of the dataset
    print("model is being evaluated on MIMIC dataset")
    results_all = pd.DataFrame(columns=["window_length", "batch_size", "loss", "accuracy", "precision", "recall", "F1"])
    # test the model on the test
    batch_size = x_l_test.shape[0]
    print(f"current testing batch size ={batch_size}")
    results = model.evaluate([x_l_test, x_d_test], y_test, batch_size=batch_size)
    results = [round(num, 2) for num in results]
    mimic_results_df = pd.DataFrame(data=[results], columns=["test loss MIMIC", "test acc MIMIC", "test prec MIMIC",
                                                             "test recall MIMIC", "test F1 MIMIC"])
    mimic_results_table = wandb.Table(dataframe=mimic_results_df)
    wandb.log({'Testing Results MIMIC': mimic_results_table})
    print("results of testing on MIMIC")
    print("test loss MIMIC, test acc MIMIC, test prec, test recall, test F1", results)
    results.insert(0, batch_size)
    results.insert(0, window_length)
    results_edited = [round(num, 2) for num in results]
    print(results_edited)
    results_all.loc[len(results_all)] = results_edited
    results_pred = model.predict([x_l_test, x_d_test], batch_size=batch_size)
    results_pred[results_pred >= f1_threshold] = 1
    results_pred[results_pred < f1_threshold] = 0

    output_columns_names.insert(len(output_columns_names), "micro avg")
    output_columns_names.insert(len(output_columns_names), "macro avg")
    output_columns_names.insert(len(output_columns_names), "weighted avg")
    output_columns_names.insert(len(output_columns_names), "samples avg")

    # cm = skm.multilabel_confusion_matrix(y_test, results_pred)
    # print("confusion matrices \n", cm)
    cr = skm.classification_report(y_test, results_pred, output_dict=True)
    detailed_results = pd.DataFrame(data=cr).transpose()
    detailed_results.reset_index(inplace=True)
    detailed_results["lab value name"] = output_columns_names
    MIMIC_results_table_detailed = wandb.Table(dataframe=detailed_results)
    wandb.log({'Detailed Results MIMIC': MIMIC_results_table_detailed})

    # print(detailed_results)
    # detailed_results.to_csv("F1_score_lab_mimic.csv", index=False)

    return results_all


def test_on_eicu(model, window_length, f1_threshold, x_l_test, x_v_test, x_d_test, y_test, output_columns_names):
    results_all = pd.DataFrame(columns=["window_length", "batch_size", "loss", "accuracy", "precision", "recall", "F1"])
    print("model is being evaluated on EICU dataset")

    batch_size = x_l_test.shape[0]
    print(f"current eicu testing batch size ={batch_size}")

    results = model.evaluate([x_l_test, x_d_test], y_test, batch_size=64)
    results = [round(num, 2) for num in results]
    eicu_results_df = pd.DataFrame(data=[results],
                                   columns=["test loss eICU", "test acc eICU", "test prec eICU", "test recall eICU",
                                            "test F1 eICU"])
    eicu_results_table = wandb.Table(dataframe=eicu_results_df)
    wandb.log({'Testing Results eICU': eicu_results_table})

    print("results of testing on eICU")
    print("test loss, test acc, test prec, test recall:", results)

    results.insert(0, batch_size)
    results.insert(0, window_length)

    results_all.loc[len(results_all)] = results
    results_pred = model.predict([x_l_test, x_d_test], batch_size=64)
    results_pred[results_pred >= f1_threshold] = 1
    results_pred[results_pred < f1_threshold] = 0

    output_columns_names.insert(len(output_columns_names), "micro avg")
    output_columns_names.insert(len(output_columns_names), "macro avg")
    output_columns_names.insert(len(output_columns_names), "weighted avg")
    output_columns_names.insert(len(output_columns_names), "samples avg")

    # cm = skm.multilabel_confusion_matrix(y_test, results_pred)
    # print("confusion matrices \n", cm)

    cr = skm.classification_report(y_test, results_pred, output_dict=True)
    detailed_results = pd.DataFrame(data=cr).transpose()
    detailed_results.reset_index(inplace=True)
    detailed_results["lab value name"] = output_columns_names
    eicu_results_table_detailed = wandb.Table(dataframe=detailed_results)
    wandb.log({'Detailed Results eicu': eicu_results_table_detailed})

    return results_all


if __name__ == '__main__':
    # ------------------  Initialization  --------------------------
    tf_gpu()
    # Initialize wandb with your project name
    run = wandb.init(project='ai_lad',
                     entity='a-ayad',
                     config={
                         "learning_rate": 0.001,
                         "epochs": 100,
                         "batch_size": 128,
                         "architecture": "TCN_3",
                         "val_split": 0.15,
                         "test_split": 0.2,
                         "F1_threshold": 0.5,
                         "dataset": "MIMIC-III",
                         "window_length": 6,
                         "feature_count": 25,
                         "model_name": "TCN_eicu_3"})

    config = wandb.config
    model_to_build = TCN_Window_DVL

    # build and compile model
    model = model_to_build.build(window_length=config.window_length, f1_threshold=config.F1_threshold,
                                 learning_rate=config.learning_rate, no_features_lab=config.feature_count,
                                 num_classes=config.feature_count)
    # we need to check here if we have already trained the model and wanted to test it only
    model.summary()

    x_l_train_m, x_l_test_m, x_v_train_m, x_v_test_m, x_d_train_m, x_d_test_m, y_train_m, y_test_m, output_columns_names_m = generate_training_data_mimic(
        window_length=config.window_length, testing_percent=(1 - config.test_split))
    x_l_train_e, x_l_test_e, x_v_train_e, x_v_test_e, x_d_train_e, x_d_test_e, y_train_e, y_test_e, output_columns_names_e = generate_training_data_eicu(
        window_length=config.window_length, testing_percent=(config.test_split))

    trained_model, training_history = train_model(model, config.epochs, config.val_split, config.batch_size,
                                                  x_l_train_e,
                                                  x_v_train_e, x_d_train_e, y_train_e)

    results_mimic = test_on_mimic(model=trained_model, window_length=config.window_length,
                                  f1_threshold=config.F1_threshold,
                                  x_l_test=x_l_test_m, x_v_test=x_v_test_m,
                                  x_d_test=x_d_test_m, y_test=y_test_m, output_columns_names=output_columns_names_m)

    results_eicu = test_on_eicu(trained_model, config.window_length, config.F1_threshold, x_l_test_e, x_v_test_e,
                                x_d_test_e, y_test_e, output_columns_names_e)
    '''
    save_model(model=trained_model, model_name=config.model_name, history=training_history, window_length=config.window_length,
               results_1=results_mimic, results_2=results_eicu)
    '''
