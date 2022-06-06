"""

* Copyright (C) RWTH Aachen University, - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""

# Assures that imports are found regardless of the base_path (Terminal) or IDE used
import sys, pathlib
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import datetime
import pandas as pd
from get_training_data_non_DL import get_training_data, get_training_data_eicu
from sklearn.utils import multiclass
import sklearn.metrics as skm
import time
# Framework imports
import lightgbm as lgb


from sklearn.model_selection import train_test_split
from sklearn import metrics



# other imports
import tensorflow as tf
import os
import numpy as np

base_path = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(base_path)
tf.keras.backend.set_floatx('float32')

# ------------------  Parameters  --------------------------


# Model

# ACTIVATION = None
# Training
VALIDATION_SPLIT = 0.15
TESTING_SPLIT = 0.2
BATCH_SIZE = 128
PRECISION_RECALL_THRESHOLD = 0.4
WINDOW_LENGTH = 5
input_lab_dimention = 25



def generate_training_data_mimic(window_length, testing_percent,lab_value_no ):
    x_l_train, x_l_test, y_train, y_test, output_columns_names = get_training_data(
        window_length=window_length, testing_percent=testing_percent,lab_value_no=lab_value_no)
    return x_l_train, x_l_test, y_train, y_test, output_columns_names




def train_model(x_l_train,x_l_test, y_train, y_test,total_time ):


    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)

    model.fit(x_l_train, y_train, eval_set=[(x_l_test, y_test), (x_l_train, y_train)],
              verbose=20, eval_metric='logloss')
    print('Training accuracy {:.4f}'.format(model.score(x_l_train, y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(x_l_test, y_test)))
    s = time.time()
    model.predict(x_l_test[:128])
    curr_time = (time.time() - s) * 1000
    total_time = total_time + curr_time
    print("current time =" , curr_time)
    return total_time





if __name__ == '__main__':
    # ------------------  Initialization  --------------------------

    # obtain paths and logdirs
    log_date = datetime.datetime.now().strftime("%Y%m%d-%H")
    log_dir = os.path.join(base_path, "data", "logs", "fit", log_date)
    F1_THRESHOLD=0.4
    # build and compile model

    total_time = 0
    # we need to check here if we have already trained the model and wanted to test it only
    results_train = pd.DataFrame(columns=["lab_id", "precision", "recall", "F1", "accuracy"])
    results_test = pd.DataFrame(columns=["lab_id","precision", "recall", "F1", "accuracy"])
    for lab_value_no in range(25):
        x_l_train, x_l_test, y_train, y_test, output_columns_names = generate_training_data_mimic(
            window_length=WINDOW_LENGTH, testing_percent=(TESTING_SPLIT), lab_value_no=lab_value_no)

        total_time= train_model(x_l_train,x_l_test, y_train, y_test,total_time)
        print("total time equals to", total_time)

