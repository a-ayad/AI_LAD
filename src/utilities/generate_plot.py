import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

# paths for files to be read
path = Path(__file__).resolve().parents[2]
tests = ["loss", "accuracy", "precision", "recall"]
model_code="CNN_LSTM"
history_file_path = os.path.join(path, "data", "model_history", f"{model_code}_f42_e300")
model_name = "CNN_LSTM"

def main():
    history_pkl = pickle.load(open(history_file_path, "rb"))
    hist_df = pd.DataFrame(history_pkl)
    print(hist_df)
    print(hist_df.columns)
    #hist_df["batch_epoch"]=hist_df["epoch"].astype(str)+"_"+hist_df["batch_number"].astype("str")
    print(hist_df.iloc[-1])
    return history_pkl


def generate_fig(history_pkl):
    # summarize history for loss
    for test in tests:
        plt.plot(history_pkl[test])
        plt.plot(history_pkl[f'val_{test}'])
        plt.title(f'{model_name} {test}')
        plt.ylabel(test)
        plt.xlabel('epochs')
        plt.legend(['train', 'val'], loc='upper left')
        plt.draw()
        fig_file_path = os.path.join(path, "data", "figures", f"{model_code}_{test}.png")
        plt.savefig(fig_file_path, dpi=300)
        plt.show()


if __name__ == "__main__":
    history_pkl = main()
    generate_fig(history_pkl)
