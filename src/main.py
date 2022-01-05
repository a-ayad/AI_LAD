# use the lstm batches
# load the model
# apply the model on the test dataset
# give results


# Standard library imports
import os
from pathlib import Path
# Third party libs imports
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.models.tf_gpu import tf_gpu

tf_gpu()
path = Path(__file__).resolve().parents[0]
# paths for patient's data
input_data_path = os.path.join(path, "data","lab_values", "input_w_lab.csv")
output_data_path = os.path.join(path, "data","lab_values", "lab_values.csv")
output_data_binary_path = os.path.join(path,"data","lab_values",  "lab_results.csv")
full_lab_names_path = os.path.join(path, "data","lab_values", "full_lab_names.csv")
raw_data_path = os.path.join(path,"data","lab_values", "raw_data.csv")




def load_model(model_name):
    model_path = os.path.join(path, "src", "models", "saved_models", model_name)
    model = tf.keras.models.load_model(model_path)
    # Show the model architecture
    model.summary()
    return model



def apply_model(model_name):
    model=load_model()



def get_testing(i):
    # read the input and output files
    # read the input and output files
    x_test = np.load(os.path.join(path, "data", "lab_values", "training_batches", f"testing_x_{i}.npy"), )
    y_test = np.load(os.path.join(path, "data", "lab_values", "training_batches", f"testing_y_{i}.npy"), )

    print("testing X equals= \n", x_test.shape)
    print("testing Y equals= \n", y_test.shape)

    return x_test,y_test


def predict_value(x_test,model):

    results_pred = model.predict(x_test, batch_size=1)
    results_pred[results_pred >= 0.2] = 1
    results_pred[results_pred < 0.2] = 0

    return  results_pred

def cal_score(y_sub,predictions_sub):
    precision=precision_score(y_true=y_sub, y_pred=predictions_sub, average='weighted')
    f1=f1_score(y_true=y_sub, y_pred=predictions_sub, average='weighted')
    recall=recall_score(y_true=y_sub, y_pred=predictions_sub, average='weighted')
    print("Precision= {} , Recall= {}, F1= {}".format(precision,recall,f1))
    return precision,recall,f1




def main(model_name):
    x_test, y_test=get_testing()
    model = load_model(model_name)
    scores=pd.DataFrame(columns=['length','precision','recall','f1'])
    for end_time_step in range(start_length,seq_length):
        x_sub,y_sub=return_subset(x_test,y_test,end_time_step)
        print("shape of y = ", y_sub.shape)
        print("shape of x= ", x_sub.shape)
        predictions=predict_value(x_sub,model)
        predictions_sub=predictions[:,-1]
        print("shape pf predictions= ", predictions_sub.shape)
        precidsion,recall,f1=cal_score(y_sub,predictions_sub)
        scores_row=[int(end_time_step),round(precidsion,2),round(recall,2),round(f1,2)]
        scores = scores.append(dict(zip(scores.columns, scores_row)), ignore_index=True)
        print(scores)

    plt.figure()
    scores.plot(x='length',y='precision',color='red',ax=ax)
    scores.plot(x='length', y='recall',color='black',ax=ax)
    scores.plot(x='length', y='f1',color='blue',ax=ax)
    plt.show()

# --------------------------------------------------------------------

if __name__ == "__main__":
    model_name="BASIC_LSTM_KT_21.h5"
    seq_length=18
    testing_size = 500
    x_test, y_test = get_testing()
    start_length=4
    main(model_name)