import pandas as pd
import os
import matplotlib.pyplot as plt
cwd=os.path.dirname(os.path.abspath(__file__))


history_file_path = os.path.join(cwd,"..",'..',"data","model_history", "BASIC_LSTM_KT_21")
history_file_path_csv=os.path.join(cwd,"..",'..',"data", "testing", "model_training_history.csv")

fig_path=os.path.join(cwd,"..",'..',"data", "figures", "model_training_fig_recall.png")

unpickled_df = pd.read_pickle(history_file_path)
unpickled_df.to_csv(history_file_path_csv, index=False)


print(unpickled_df.columns)

mean_col_loss=unpickled_df.groupby("epoch")["loss"].mean().reset_index()
mean_col_accuracy=unpickled_df.groupby("epoch")["val_accuracy"].mean().reset_index()
mean_col_precision=unpickled_df.groupby("epoch")["val_precision"].mean().reset_index()
mean_col_recall=unpickled_df.groupby("epoch")["val_recall"].mean().reset_index()




mean_col_recall.plot(x='epoch',y=['val_recall'],title="Model Recall")
plt.savefig(fig_path)
#plt.show()


