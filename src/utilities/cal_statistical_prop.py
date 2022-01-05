import pandas as pd
import os


cwd=os.path.dirname(os.path.abspath(__file__))

# input and output files loaded:
input_data_path = os.path.join(cwd,"..",'..', "data","lab_values", "input_w_lab.csv")
statistical_df_data_path = os.path.join(cwd,"..",'..',"data", "lab_values", "mimic_statistical.csv")


lab_values=pd.read_csv(input_data_path)
print(lab_values)
lab_values.drop("bloc",axis=1,inplace=True)
lab_names=lab_values.columns.tolist()
lab_names.remove("icustayid")

grouped_mean=lab_values.groupby("icustayid").mean()
grouped_max=lab_values.groupby("icustayid").max()
grouped_min=lab_values.groupby("icustayid").min()
grouped_std=lab_values.groupby("icustayid").std()


means_mean=grouped_mean.mean()
means_std=grouped_std.mean()
means_max=grouped_max.max()
means_min=grouped_min.min()
statistical_df = pd.DataFrame(columns=["feature_name", "mean", "std", "max", "min"])
statistical_df["feature_name"]=lab_names
statistical_df["mean"]=means_mean.tolist()
statistical_df["std"]=means_std.tolist()
statistical_df["max"]=means_max.tolist()
statistical_df["min"]=means_min.tolist()
statistical_df=round(statistical_df,2)
print(statistical_df)

statistical_df.to_csv(statistical_df_data_path,index=False)


