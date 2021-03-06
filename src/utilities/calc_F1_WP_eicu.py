
# Standard library imports
from pathlib import Path
import os
# Third party libs imports
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics as skm
from sklearn.utils import multiclass
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN, tcn_full_summary
path = Path(__file__).resolve().parents[2]

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










def load_model(model_name):
    model_path=os.path.join(path,"src","models","saved_models",model_name)
    model = tf.keras.models.load_model(model_path)
    # ,custom_objects = {'TCN': TCN}
    # Show the model architecture
    model.summary()
    return model


def predict_value(window_length, model_name, f1_threshold):
    results_df = pd.DataFrame(columns=["window_length", "precision", "recall", "f1-score", "support"])
    results_all = pd.DataFrame(columns=["window_length", "batch_size", "loss", "accuracy", "precision", "recall"])
    model = load_model(model_name)
    print("model is loaded")
    print(os.getcwd())
    train_x_1 = np.load(
        os.path.join(path, "data", "lab_values", "windowed_data_eicu", f"test_x_1_WL{window_length}.npy"),
        allow_pickle=True)
    train_x_2 = np.load(
        os.path.join(path, "data", "lab_values", "windowed_data_eicu", f"test_x_2_WL{window_length}.npy"),
        allow_pickle=True)
    train_y = np.load(
        os.path.join(path, "data", "lab_values", "windowed_data_eicu", f"test_y_WL{window_length}.npy"),
        allow_pickle=True)
    # test the model on the test
    # test the model on the test
    tf_gpu()
    train_x_1 = np.asarray(train_x_1).astype('float32')
    train_x_2 = np.asarray(train_x_2).astype('float32')
    train_y = np.asarray(train_y).astype('float32')
    batch_size = train_x_1.shape[0]

    print(f"current batch size ={batch_size}")

    results = model.evaluate([train_x_1, train_x_2], train_y, batch_size=64)
    print("test loss, test acc, test prec, test recall:", results)
    '''v
    results = [round(num, 2) for num in results]
    results.insert(0, batch_size)
    results.insert(0, length)
    results_all.loc[len(results_all)] = results
    last_time_step_each_input_sequence = x_test[:, -1]
    last_time_step_each_output_sequence = y_test[:, -1]
    print(f"shape of last time step of each input sequence= {last_time_step_each_input_sequence.shape}")
    print(f"shape of last time step of each output sequence= {last_time_step_each_output_sequence.shape}")
    results_pred = model.predict(x_test, batch_size=batch_size)
    results_pred[results_pred >= f1_threshold] = 1
    results_pred[results_pred < f1_threshold] = 0
    results_pred_last = results_pred[:, -1]
    # print(results_pred_last)
    print(results_pred_last.shape)
    # print(results_pred_last)
    cm = skm.multilabel_confusion_matrix(last_time_step_each_output_sequence, results_pred_last)
    # print("confusion matrices \n", cm)
    cr = skm.classification_report(last_time_step_each_output_sequence, results_pred_last, output_dict=True)
    df = pd.DataFrame(data=cr).transpose()

    df.to_csv(os.path.join(path, "data", "f1_results", f"{model_name}_L{length}_last_output.csv"))
    scores_list = df.loc["weighted avg"].tolist()
    scores_list.insert(0, length)
    scores_list_edited = [round(num, 2) for num in scores_list]
    print(scores_list_edited)
    results_df.loc[len(results_df)] = scores_list_edited
            
    #results_df.drop(4, axis=0,inplace=True)
    results_df.loc['mean'] = results_df.mean()
    results_all.loc['mean'] = results_all.mean()
    print(results_df)
    print(results_all)
    results_df.to_csv(os.path.join(path, "data", "f1_results", f"{model_name}_L{length}_last_output_avg.csv"))
    results_all.to_csv(os.path.join(path, "data", "f1_results", f"{model_name}_L{length}_all_output_avg.csv"))
# --------------------------------------------------------------------
    '''
if __name__ == "__main__":
    model_name="CNN_WP_f37_e300_WL5.h5"
    window_length=5
    f1_threshold = 0.4
    predict_value(window_length, model_name, f1_threshold)


